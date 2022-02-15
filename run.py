# Imports -----------------------------------------------------------------------------------------------------------------------
from flask import Flask, request, jsonify, send_file
import os,binascii
import time
import threading
import os
import numpy as np
import cv2
import time
import sys
import torch
import torchvision.transforms as transforms
import tensorflow as tf
import clip
import pickle
import copy
import random
import urllib.request
from argparse import Namespace
import matplotlib.pyplot as plt
from PIL import Image
from interfacegan.models.stylegan_generator import StyleGANGenerator
from interfacegan.utils.manipulator import linear_interpolate
from encoder4editing.utils.common import tensor2im
from encoder4editing.models.psp import pSp
from StyleCLIP.global_directions.MapTS import GetFs,GetBoundary,GetDt
from StyleCLIP.global_directions.manipulate import Manipulator
from BG_Remover.inference import BGRemove
import dnnlib
import dnnlib.tflib as tflib
import dlib
from encoder4editing.utils.alignment import align_face
from FaceSwapX import main

app = Flask(__name__)
SESSION_LIMIT = 1800
SESSION_QUEUE = {}


@app.route('/')
def index():
    return 'Welcome to DSMATICS!'

def generate_session_id():
    return binascii.b2a_hex(os.urandom(8)).decode("utf-8") 

def clean_up(id):
  for file in os.listdir('encoded_output'):
    idx = file[:-4].split('_')[-1]
    if(idx==id):
      os.remove(f'encoded_output/{file}')

@app.route('/login', methods = ['GET'])
def login():
    temp_id = generate_session_id()
    SESSION_QUEUE[temp_id]=time.time()
    print("Session Created:",temp_id)
    return jsonify({"id":temp_id})

def maintainQueue():
    while True:
        dispose_=[]
        for key in SESSION_QUEUE:
            if time.time()-SESSION_QUEUE[key]>SESSION_LIMIT:
                dispose_.append(key)
        for key in dispose_:
            del SESSION_QUEUE[key]
            clean_up(key)
            print("Session Deleted:",key)
        time.sleep(5)

def is_valid_session(id):
    if id in SESSION_QUEUE:
        return True
    return False

threading.Thread(target=maintainQueue).start()

# Methods -----------------------------------------------------------------------------------------------------------------------

def build_generator(model_name):
  generator = StyleGANGenerator(model_name)
  return generator

def run_alignment(image_path):
  predictor = dlib.shape_predictor("encoder4editing/shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor) 
  return aligned_image

def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return images, latents

def load_boundaries(attrs_names):
  boundaries = {}
  for i, attr_name in enumerate(attrs_names):
      boundary_name = f'stylegan_ffhq_{attr_name}'
      boundaries[attr_name] = np.load(f'interfacegan/boundaries/{boundary_name}_boundary.npy')
  return boundaries

def load_latent_codes_for_stylegan(path,filter='r'):
  latent_codes = pickle.load(open(path, 'rb'))
  if(filter=='r'):
    op = random.randint(0,1)
    if op:
      filter='m'
    else:
      filter = 'f'
  
  idx = random.randint(0,latent_codes[filter].shape[0]-1)
  latent_code = latent_codes[filter][idx,0,:].reshape((1,512))
  return latent_code

def morph_images_with_stylegan(generator,new_codes,attrs_names,attr_values,boundaries,synthesis_kwargs):
  for attr_name in attrs_names:
    new_codes += boundaries[attr_name] * attr_values[attr_name]
  new_images = generator.easy_synthesize(new_codes, **synthesis_kwargs)['image']
  return new_codes,new_images

def synthesise_image_from_latent_codes(generator,latent_codes,synthesis_kwargs):
  images = generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image']
  return images

def save_image(image,path,name,ext,type,id):
  if(type==1):
    for img in image:
        Image.fromarray(img).save(f"{path}/{name}_{id}.{ext}")
  elif(type==2):
    image.save(f"{path}/{name}_{id}.{ext}")
  elif(type==3):
    alpha, img = image
    w, h, _ = img.shape
    png_image = np.zeros((w, h, 4))
    png_image[:, :, :3] = img
    png_image[:, :, 3] = alpha
    cv2.imwrite(f"{path}/{name}_{id}.{ext}", png_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
  elif(type==4):
    cv2.imwrite(f"{path}/{name}_{id}.{ext}", image)

def encode_image_for_stylegan2(image_path):
  original_image = Image.open(image_path)
  original_image = original_image.convert("RGB")
  input_image = run_alignment(image_path)
  input_image.resize(resize_dims)
  img_transforms = EXPERIMENT_ARGS['transform']
  transformed_image = img_transforms(input_image)
  with torch.no_grad():
      images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
      result_image, latent = images[0], latents[0]
  return latents, images

def change_head_pitch(latent_representation, Gs_network, latent_weight=10):
  latent_direction = np.expand_dims(np.load('pitch.npy', allow_pickle=True),axis=0)
  new_code = latent_representation + (latent_direction * latent_weight)
  tflib.init_tf()
  generator_network, discriminator_network, Gs_network = pickle.load(open('stylegan2-ffhq-config-f.pkl','rb'))
  synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
  images = Gs_network.components.synthesis.run(new_code, randomize_noise=False, **synthesis_kwargs)
  return new_code,images

def morph_face_with_styleclip(M,neutral,target,model,latent_vectors,alpha,beta,is_styleclip):
  M.num_images=1
  M.alpha=[0]
  M.manipulate_layers=[0]
  M.manipulate_layers=None
  classnames=[target,neutral]
  dt=GetDt(classnames,model)
  if is_styleclip:
    dlatent_tmp = latent_vectors
  else:
    dlatent_tmp = M.W2S(latent_vectors)
  M.alpha=[alpha]
  boundary_tmp2,c=GetBoundary(fs3,dt,M,threshold=beta)
  codes=M.MSCode(dlatent_tmp,boundary_tmp2)
  out=M.GenerateImg(codes)
  generated=Image.fromarray(out[0,0])
  for idx in range(len(dlatent_tmp)):
    codes[idx] = np.squeeze(codes[idx], axis=0)
  return generated,codes

def remove_background(path,output='BG_Remover/output/'):
  try:
    matte = bg_remover.image(path, background=False, output=output, save=False)
    return matte
  except Exception as Err:
    return f"Erro happend {Err}"

def getLatentVectors(id,on_original=False):
  if on_original:
    latents,images, = encode_image_for_stylegan2(f'encoded_output/original_image_{id}.png')
  else:
    latents,images, = encode_image_for_stylegan2(f'encoded_output/encoded_image_{id}.png')
  latent_representation = latents.cpu().detach().numpy()
  return latent_representation

attr_names = [i[14:-13] for i in os.listdir('interfacegan/boundaries') if i !='.ipynb_checkpoints']
boundaries = load_boundaries(attr_names)
stylegan_generator = build_generator("stylegan_ffhq")

bg_remover = BGRemove('BG_Remover/pretrained/modnet_photographic_portrait_matting.ckpt')

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B-32.pt", device=device) 
M=Manipulator(dataset_name='ffhq')
fs3=np.load('StyleCLIP/global_directions/npy/ffhq/fs3.npy')
np.set_printoptions(suppress=True)
EXPERIMENT_ARGS = {
        "model_path": "encoder4editing/e4e_ffhq_encode.pt"
    }
EXPERIMENT_ARGS['transform'] = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
resize_dims = (256, 256)
model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts= Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()

tflib.init_tf()
generator_network, discriminator_network, Gs_network = pickle.load(open('stylegan2-ffhq-config-f.pkl','rb'))

def faceEdit_stylegan(attr_values,stylegan_generator,boundaries,latent_codes):
  synthesis_kwargs_ = {'latent_space_type': 'W'}
  latent_codes_new_images ,new_images = morph_images_with_stylegan(stylegan_generator,latent_codes,attr_names,attr_values,boundaries,synthesis_kwargs_)
  return new_images

def clean_up(id):
  for file in os.listdir('encoded_output'):
    idx = file[:-4].split('_')[-1]
    if(idx==id):
      os.remove(f'encoded_output/{file}')

# API Calls ------------------------------------------------------------------------------------------------------------------

@app.route('/generateRandomFace' , methods = ['POST'])
def generateRandomFace():
    id = request.json['id']
    filter = request.json['filter']
    if is_valid_session(id):
        clean_up(id)
        latent_codes = load_latent_codes_for_stylegan('latent_representations.pkl',filter=filter)
        images = synthesise_image_from_latent_codes(stylegan_generator,latent_codes,{'latent_space_type': 'W'})
        save_image(images,'encoded_output','original_image','png',1,id)
        pickle.dump(latent_codes,open(f'encoded_output/latent_code_{id}.pkl','wb'))
        return jsonify({"status": "Success"})
    else:
        return "Session Timed Out!"

@app.route('/globalEdit' , methods = ['POST'])
def globalEdit():
    id = request.json['id']
    attr_values = request.json['attr_values']
    if is_valid_session(id):
        latent_codes = pickle.load(open(f'encoded_output/latent_code_{id}.pkl','rb'))
        new_images = faceEdit_stylegan(attr_values,stylegan_generator,boundaries,latent_codes)
        save_image(new_images,'encoded_output','encoded_image','png',1,id)
        return jsonify({"status": "Success"})
    else:
        return "Session Timed Out!"


@app.route('/pitchEdit' , methods = ['POST'])
def pitchEdit():
    id = request.json['id']
    latent_weight = request.json['latent_weight']
    on_original = request.json['on_original']
    if is_valid_session(id):
        latent_representation = getLatentVectors(id,on_original=on_original)
        tflib.init_tf()
        latent_codes_pitched, image_with_pitch = change_head_pitch(latent_representation, Gs_network, latent_weight=latent_weight)
        save_image(Image.fromarray(image_with_pitch.transpose((0,2,3,1))[0], 'RGB'),'encoded_output','encoded_image','png',2,id)
        pickle.dump(latent_codes_pitched,open(f'encoded_output/latent_representation_{id}.pkl','wb'))
        return jsonify({"status": "Success"})
    else:
        return "Session Timed Out!"

@app.route('/localEdit' , methods = ['POST'])
def localEdit():
    id = request.json['id']
    is_styleclip = request.json['is_styleclip']
    on_original = request.json['on_original']
    alpha = request.json['alpha']
    target_ = request.json['target_']
    style_idx = request.json['style_idx']
    if is_valid_session(id):
        path = f'encoded_output/latent_representation_{id}.pkl'
        if os.path.isfile(path):
          if on_original:
            latent_representation = getLatentVectors(id,on_original=on_original)
          else:
            latent_representation = pickle.load(open(path,'rb'))
        else:
          latent_representation = getLatentVectors(id,on_original=on_original)
        beta = 0.15 #min:0.08, max:0.3, step:0.01}
        target_style = {'Ethenicity':['Australian','Indian','Asian','Latino','Hawaiian','Blonde'],
        'Hair Color': ['Black Hair Color', 'Brown Hair Color', 'Blonde Hair Color', 'Red Hair Color', 'White Hair Color', 'Silver Hair Color'],
        'Hair Highlights': ['Black Hair Highlights', 'brown Hair Highlights', 'Blonde Hair Highlights', 'Red Hair Highlights', 'White Hair Highlights', 'Silver Hair Highlights', 'golden Hair Color'],
        'Hairstyle': ['Bald', 'Bangs', 'Curly Hair', 'Hi-top Fade', 'Fringe Hair', 'Bob Cut', 'Mohawk'],
        'Facial Hair': ['Moustache', 'Beard'],
        'Accesories': ['Earrings'],
        'Emotions': ['Surprised Face', 'Angry Face', 'Contemptuous Face', 'Sad Face', 'Happy Face', 'Disgusted Face'],
        'Eye Color': ['Blue Eyes'],
        'Eye Size': ['Big Eyes'],
        'Face Shape': ['Fat Face', 'Chubby Face'],
        'Makeup': ['Eye Liner', 'Eye Makeup', 'Lips Makeup', 'Eyebrow Makeup'],
        'Ear Size': ['Large Ears'],
        'Lip Size': ['Large Lips'],
        'Facespots': ['Freckles']}
        neutral='face'
        tflib.init_tf()
        M=Manipulator(dataset_name='ffhq')
        generated,dlatents= None,None
        for alphax, targetx, style_idxx in zip(alpha,target_,style_idx):
          target=target_style[targetx][style_idxx]
          generated,dlatents = morph_face_with_styleclip(M, neutral, target, model, latent_representation, alphax, beta, is_styleclip)
          latent_representation = dlatents
          is_styleclip = True
        save_image(generated,'encoded_output','encoded_image','png',2, id)
        pickle.dump(dlatents,open(f'encoded_output/latent_representation_{id}.pkl','wb'))
        return jsonify({"status": "Success"})
    else:
        return "Session Timed Out!"

@app.route('/removeBackground' , methods = ['POST'])
def removeBackground():
    id = request.json['id']
    image_url = request.json['image_url']
    if is_valid_session(id):
      filename = f'encoded_output/srcImage_{id}{image_url[-4:]}'
      try:
        urllib.request.urlretrieve(image_url, filename)
        matte = remove_background(filename,'encoded_output/')
        save_image(matte,'encoded_output/', f'encoded_image' ,'png',3, id)
        return jsonify({"status": "Success"})
      except:
        return jsonify({"status": "Error: Unable to access the image"})
    else:
        return "Session Timed Out!"

@app.route('/faceSwap' , methods = ['POST'])
def faceSwap():
    id = request.json['id']
    src_image_url = request.json['src_image_url']
    dst_image_url = request.json['dst_image_url']
    if is_valid_session(id):
      src_filename = f'encoded_output/srcImage_{id}{src_image_url[-4:]}'
      dst_filename = f'encoded_output/dstImage_{id}{dst_image_url[-4:]}'
      try:
        urllib.request.urlretrieve(src_image_url, src_filename)
        urllib.request.urlretrieve(dst_image_url, dst_filename)
        img = main.swap_faces(src_filename,dst_filename)
        save_image(img,'encoded_output/', f'encoded_image' ,'png',4, id)
        return jsonify({"status": "Success"})
      except:
        return jsonify({"status": "Error: Unable to access the image"})
    else:
        return "Session Timed Out!"

@app.route('/getImage' , methods = ['POST'])
def getImage():
    id = request.json['id']
    get_original = request.json['get_original']
    if is_valid_session(id):
        if get_original:
          path = f'encoded_output/original_image_{id}.png'
        else: 
          path = f'encoded_output/encoded_image_{id}.png'
        if os.path.isfile(path):
            return send_file(path, as_attachment=True)
        else:
            return "Error: File not found!"
    else:
        return "Session Timed Out!"

if __name__ == '__main__':
    app.run(host='0.0.0.0')