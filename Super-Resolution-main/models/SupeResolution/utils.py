import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from torchvision import transforms
from model.person_ext.rvm.model import MattingNetwork
from torch.utils.data import DataLoader
from model.person_ext.rvm.inference_utils import VideoReader
import pickle
from model.person_cls.classification import Classification
from PIL import Image
import numpy as np
from numpy.linalg import norm
import cv2
from scipy.stats import logistic, johnsonsb, t, burr
import mediapipe as mp

import numpy as np
from model.gait.opengait.tools import config_loader, params_count, get_msg_mgr
from model.gait.opengait.modeling import models
import datetime
import glob
import json

from insightface.app import FaceAnalysis

# Load face analysis model
app = FaceAnalysis(name='thecircleface', root="~/.insightface", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Global variables
# ============================================================
# Load all models
classfication = Classification()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# gait_model = None

# Functions 
# ============================================================

def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)

# =========================

def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).to(device)
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin

# ===============================
def calc_same_group_probability(x):
    # HID500-OutdoorGait138
    args = {'df': 4.99535496370145, 'loc': 9.62629510552382, 'scale': 0.915049924855037}
    res = t.cdf(x=x, **args)

    return res


def calc_diff_group_probability(x):
    # HID500-OutdoorGait138
    args = {'a': 22.195523809517333, 'b': 27.46516021982316, 'loc': -48.1526371634257, 'scale': 205.14459476282795}
    res = johnsonsb.cdf(x=x, **args)

    return res


def calc_similarity(dist, w1=1.0, w2=0):
    if abs(w1 + w2 - 1.0) > 0.01:
        print("Error! The sum of w1 and w2 must be 1.0")
        return None
    a = calc_same_group_probability(x=dist)
    b = calc_diff_group_probability(x=dist)
    similarity = w1 * (1-a) + w2 * (1-b)
    return similarity

def cosine_similarity(a, b):
    """
    compute the cosine similarity between face embeddings a and b
    used for face embeddings. a and b are np arrays
    """
    # flatten the embeddings
    a = a.ravel()
    b = b.ravel()
    
    similarity = np.dot(a, b) / (norm(a) * norm(b))
    return similarity


def compare_face(face_embedding):
    """
    compare face embedding to the enrolled face embeddings
    """
    db_files = [file for file in os.listdir('enrolled') if file.endswith('.json')]
    
    if db_files == []:
        return
    
    # check if the data in the db has an associated face embedding with it
    db_with_embeddings = []
    for json_file in db_files:
        with open(f"enrolled/{json_file}", 'r') as file:
            data = json.load(file)
            
            if "face_embedding" in data:
                embedding_dict = {}
                embedding_dict["db_data"] = data
                embedding_dict["np_embedding"] = np.load(data["face_embedding"])
                db_with_embeddings.append(embedding_dict)
    if db_with_embeddings == []:
        return
    
    similarities = []
    for embedding_dict in db_with_embeddings:
        similarities.append(cosine_similarity(face_embedding, embedding_dict["np_embedding"]))
    # find the index of the maximum similarity
    index = np.argmax( np.array(similarities) )
    
    # only consider a face match if the cosine similarity is greater than 0.6
    if similarities[index] > 0.6:
        # return the data of the matched person and the cosine similarity
        return db_with_embeddings[index]["db_data"], similarities[index]

# ==============================================

def is_people(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    class_name, probability = classfication.detect_image(img)
    if class_name == 'people' and probability > 0.5:
        return True
    else:
        return False

# ===============================================

def cut_img(img, img_size, frame_name, pixel_threshold=0):
    # A silhouette contains too little white pixels
    # might be not valid for identification.
    if img is None or img.sum() <= 10000:
#         print(f'\t {frame_name} has no data. {img.sum()}')
        return None

    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y > pixel_threshold).argmax(axis=0)  # the line pixels more than pixel_threshold, it will be counted
    y_btm = (y > pixel_threshold).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]

    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)

    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
#         print(f'\t{frame_name} has no center.')
        return None

    # Get the left and right points
    half_width = img_size // 2
    left = x_center - half_width
    right = x_center + half_width
    if left <= 0 or right >= img.shape[1]:
        left += half_width
        right += half_width
        _ = np.zeros((img.shape[0], half_width))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right].astype('uint8')
    return img

# ===================================================================

def load_humanseg_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load model
    # ==========================
    print('Loading human segmentation model...')
    # load mobilenetv3 model
    model = MattingNetwork('mobilenetv3').eval().to(device)
    model_path = 'model/person_ext/rvm/work/checkpoint/rvm_mobilenetv3.pth'
    model.load_state_dict(torch.load(model_path))

    # Inference preparation
    dtype = None
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device

    return model, dtype, device

def get_silhouettes(vidsource, personid):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models 
    # ==========================
    print('Loading human segmentation model...')
    # load mobilenetv3 model
    model = MattingNetwork('mobilenetv3').eval().to(device)
    model_path = 'model/person_ext/rvm/work/checkpoint/rvm_mobilenetv3.pth'
    model.load_state_dict(torch.load(model_path))

    # Inference preparation
    dtype = None
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device

    transform = transforms.ToTensor()

    # input_source = 'person15_jogging_d1_uncomp.avi'
    input_source = vidsource

    print('Creating video capture object')
    reader = cv2.VideoCapture(input_source)

    pixel_threshold = 800
    img_size = 64
    silhouettes = []

    print('Processing stream....')
    with torch.no_grad():
        rec = [None] * 4
        while reader.isOpened():
            ret, src = reader.read()
            if ret:
                downsample_ratio = auto_downsample_ratio(*src.shape[:-1])
                if src.shape[0] > 540:
                    src = cv2.resize(src, (960, 540), interpolation=cv2.INTER_CUBIC)
                src = transform(src)
                src = src.to(device, dtype, non_blocking=True).unsqueeze(0)  # [B, T, C, H, W]
                fgr, pha, *rec = model(src, *rec, downsample_ratio)
                fgr = fgr * pha.gt(0)
                com = torch.cat([fgr, pha], dim=-3)

                im = (com[0][3].cpu().numpy()*255).astype('uint8')
                img = cut_img(im, 128, 0, pixel_threshold)

                if img is None:
                    continue
                if is_people(img):
                    # resize
                    ratio = img.shape[1] / img.shape[0]
                    img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)
                    silhouettes.append(img)

            else:
                break

    # release the video capture object
    reader.release()
    # Closes all the windows currently opened.
    # cv2.destroyAllWindows()

    silhouettes = np.array(silhouettes)

    # Save the silhouettes
    pickle.dump(silhouettes, open(f'{personid}-silhouettes.pkl', 'wb'))

    # clear the model from memory
    del model

    return silhouettes

# ==============================================================
def initialization(cfgs):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])

    msg_mgr.init_logger(output_path, False) # False: not log to file
    msg_mgr.log_info(engine_cfg)

def load_gait_model():
    global gait_model
    # Load and initialize config
    cfgs = "model/gait/configs/gaitgl/gaitgl_HID_OutdoorGait_CASIA-B_OUMVLP.yaml"

    print('Loading gait recognition model....')
    cfgs = config_loader(cfgs)
    initialization(cfgs)

    # Load model for inference
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    gait_model = Model(cfgs, training=False)
    if cfgs['trainer_cfg']['fix_BN']:
        gait_model.fix_BN()
    msg_mgr.log_info(params_count(gait_model))
    msg_mgr.log_info("Model Initialization Finished!")
    gait_model.to(device)

    return gait_model

def get_gait_embedding(sil, gait_model):
    # global gait_model
    data = sil

    # Prepare input 
    inp = [[[data]], [0], ['probe'], ['default'], np.array([[data.shape[0]]])]

    # Pre-treatment
    ipt = gait_model.inputs_pretreament(inp)

    # inference
    retval = gait_model.forward(ipt)

    # Extract embeddings
    feat = retval['inference_feat']['embeddings'].cpu().detach().numpy()

    return feat

# ==============================================================
def extract_face(photo):
    # Extract face using insightface
    face_dict = app.get(photo, max_num=1, skip=False)

    # Extract face
    if len(face_dict) == 0:
        print('No face detected')
        return None, None

    else:
        face = face_dict[0]
        emb = face['embedding']
        bbox = face.bbox.astype('int').flatten()
        face = photo[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        return face, emb
    
    # mp_face_detection = mp.solutions.face_detection

    # # For static images:
    # face_detection = mp_face_detection.FaceDetection(
    #     model_selection=0, min_detection_confidence=0.5)
    
    # # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    # results = face_detection.process(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))

    # # Extract face region from the image
    # if not results.detections:
    #     print('No face detected')
    #     return None
    
    # for landmark in results.detections:
    #     # print(landmark.location_data.relative_bounding_box)
    #     bbox = landmark.location_data.relative_bounding_box
    #     h, w, c = photo.shape
    #     # print(bbox.xmin, bbox.ymin, bbox.width, bbox.height)
    #     xmin = int(bbox.xmin * w)
    #     ymin = int(bbox.ymin * h)
    #     xmax = int((bbox.xmin + bbox.width) * w)
    #     ymax = int((bbox.ymin + bbox.height) * h)
    #     face = photo[ymin:ymax, xmin:xmax]
    #     # print(face.shape)

# ==============================================================

def register_gait(vid, personid, pic, status, tk, gait_model):
    # global gait_model
    # load_gait_model()
    # get silhouettes
    status.configure(text='Getting silhouettes...')
    tk.update()
    sil = get_silhouettes(vid, 'person15')

    num_sil = sil.shape[0]

    # get embeddings from the gait model
    status.configure(text='Getting embeddings...')
    tk.update()
    feat = get_gait_embedding(sil, gait_model)

    # Read pic, detect face in it and resize to 160x160
    photo = cv2.imread(pic)
    face_photo, face_emb = extract_face(photo)
    # face_photo = cv2.resize(face_photo, (160, 160))

    # Save the embeddings to the folder bio-db along with the personid as a dictionary in a pickle file
    # get the current timestamp for appending to filename
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    pickle.dump({'id':personid, 'gaitemb': feat, 'face': face_photo, 'face_emb': face_emb}, open(f'bio-db/{personid}-emb-{timestamp}.pkl', 'wb'))

    print(f'Gait for {personid} registered successfully. Representation from {num_sil} silhouettes saved to bio-db/{personid}-emb-{timestamp}.pkl')
    optim_num_sil = 250
    rep_qual = num_sil/optim_num_sil * 100 if num_sil <= 100 else 100
    status.configure(text=f'Gait for {personid} registered successfully using {num_sil} silhouettes. Representation quality is rated as: {rep_qual}%')
    tk.update()

# ====================================
def compare_gait(vid, personid, gait_model):
    # get silhouettes
    sil = get_silhouettes(vid, personid)

    # get embeddings from the gait model
    probe_feat = get_gait_embedding(sil, gait_model)

    # Load the embeddings from the bio-db folder
    dblist = glob.glob('bio-db/*.pkl')
    results = []
    for dbfile in dblist:
        db = pickle.load(open(dbfile, 'rb'))
        pid = db['id']
        gaitemb = db['gaitemb']

        # Compare the embeddings
        dist = cuda_dist(probe_feat, gaitemb, 'euc').cpu().numpy()
        simi = calc_similarity(dist)
        
        results.append({'id': pid, 'dist': dist, 'similarity': simi})

        # Sort the results in descending order of similarity
        results = sorted(results, key=lambda k: k['similarity'], reverse=True)

    # Print the results
    for result in results:
        pid = result['id']
        dist = result['dist'][0][0]
        simi = result['similarity'][0][0]
        # print(f'ID: {pid}, Dist: {dist:02.3f}, Sim: {simi:02.3f}')


def compare_gait_db(probe_feat):
    
    # Load the embeddings from the bio-db folder
    dblist = glob.glob('bio-db/*.pkl')
    
    results = []
    for dbfile in dblist:
        db = pickle.load(open(dbfile, 'rb'))
        pid = db['id']
        gaitemb = db['gaitemb']
        pface = db['face']

        # Compare the embeddings
        dist = cuda_dist(probe_feat, gaitemb, 'euc').cpu().numpy()
        simi = calc_similarity(dist)
        
        results.append({'id': pid, 'dist': dist, 'similarity': simi, 'face': pface})

        # Sort the results in descending order of similarity
        results = sorted(results, key=lambda k: k['similarity'], reverse=True)

    # Print the results
    res = []
    for result in results:
        pid = result['id']
        dist = result['dist'][0][0]
        simi = result['similarity'][0][0]
        face = result['face']

        res.append([pid, dist, simi, face])
        print(f'Gait ID: {pid}, Dist: {dist:02.3f}, Sim: {simi:02.3f}')

    return res

# ==========================================

def compute_sim(feat1, feat2):
        
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

# ===========================================

def crop_face(img, face_detection_result):
    """
    img: numpy image
    face_detection_result: result from running face detection algorithm
    
    crops out the face from the image and returns it
    image is a numpy array
    """
    bbox = [int(i) for i in face_detection_result['bbox']]
    
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def compare_face_emb(face_data, photo):
    
    # Load the embeddings from the bio-db folder
    dblist = glob.glob('bio-db/*.pkl')
    results = []
    if len(face_data)==0 or face_data is None:
        return [['unknown', 0, 0, None]]
    
    # if no enrollments exist
    if(len(dblist)==0):
        pid = 'unknown'
        dist = 0
        simi = 0
        
        face = face_data[0]
        bbox = face.bbox.astype('int').flatten()
        kps = face['kps']
        # if(len(kps)>0):
        #     for i in range(0, len(kps)):
        #         cv2.circle(photo, (int(kps[i][0]), int(kps[i][1])), 2, (255, 0, 0), 1)

        face = photo[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        if face.shape[0] != 0 and face.shape[1] != 0:
            pface = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        else:
            pface = None

        face = pface

        return [[pid, dist, simi, face]]
    
    # if enrollments exist
    for dbfile in dblist:
        db = pickle.load(open(dbfile, 'rb'))
        pid = db['id']
        db_femb = db['face_emb']
        
        # extract face from the face_data

        if(face_data is None or len(face_data)==0):
            return [['unknown', 0, 0, None]]
        
        face = face_data[0]
        emb = face['embedding']
        bbox = face.bbox.astype('int').flatten()
        kps = face['kps']
        # if(len(kps)>0):
        #     for i in range(0, len(kps)):
        #         cv2.circle(photo, (int(kps[i][0]), int(kps[i][1])), 2, (255, 0, 0), 1)

        face = photo[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        if face.shape[0] != 0 and face.shape[1] != 0:
            pface = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        else:
            pface = None

        # Compare the embeddings
        if(emb is not None and db_femb is not None):
            qf1emb = emb
            qf2emb = db_femb
            simi = compute_sim( np.array(qf1emb), np.array(qf2emb) )
            dist = 1 - simi
            
            # scale simscore to above 0.75 if sim > 0.22
            if(simi >= 0.2):
                simi = (simi - 0.2) / (1.0 - 0.2)
                simi = simi * (1.0 - 0.75) + 0.75
                simi = round(simi, 2)
            else:
                # scale the similarity score <0.2 to be in the range [0.0, 0.75]
                simi = (simi - 0.0) / (0.2 - 0.0)
                simi = simi * (0.75 - 0.0) + 0.0
                simi = round(simi, 2)

            results.append({'id': pid, 'dist': dist, 'similarity': simi, 'face': pface})

    # Sort the results in descending order of similarity
    results = sorted(results, key=lambda k: k['similarity'], reverse=True)


    # Print the results
    res = []
    for result in results:
        pid = result['id']
        dist = result['dist']
        simi = result['similarity']
        face = result['face']

        res.append([pid, dist, simi, face])
        # print(f'Face ID: {pid}, Dist: {dist:02.3f}, Sim: {simi:02.3f}')

    return res


def save_video(person_frames, vfname):

    # save the video itself
    height,width,_=person_frames[1].shape
    fps = 30
    video=cv2.VideoWriter(vfname,cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    
    for j in range(len(person_frames)):
        video.write(person_frames[j])
    video.release()

# ===========================================
def update_rec_icons(gr, recdata):
    no_of_rec = min(5, len(recdata))

    recdata = recdata[::-1]
    
    for rec in range(no_of_rec):
        rd = recdata[rec]
        gr[rec].update_icon(rd['pface'], rd['sildisp'],  rd['caption'], rd['color'])

# ===================================================================
# Main
# ===================================================================
# if __name__ == '__main__':
#     # Load the gait model
#     gait_model = load_gait_model()

#     # Gait processing
#     personid = 'probe'
#     vid = 'person04_walking_d4_uncomp.avi'
#     # register_gait(vid, personid)
#     # vid = 'person05_walking_d2_uncomp.avi'
#     # register_gait(vid, personid)
#     # vid = 'person05_walking_d3_uncomp.avi'
#     # register_gait(vid, personid, gait_model)
    
#     compare_gait(vid, personid, gait_model)

    
