from django.shortcuts import render
from django.views.generic.edit import CreateView, DeleteView, UpdateView
from django.shortcuts import redirect
from django.http import HttpResponse
from .models import Photo

import os
import sys
import scipy.io
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import nst_utils as Nu
import numpy as np
import tensorflow as tf
import math

from django.conf import settings
m_u=settings.MEDIA_ROOT
s_u=settings.STATICFILES_DIRS[0]
#------------------------------------------------------------------------------#

class PhotoUploadView(CreateView):

    model = Photo
    fields = ['photo']
    template_name = 'image_upload/upload.html'

    def form_valid(self, form):
        Photo.objects.all().delete()
        form.instance.author_id = self.request.user.id
        if form.is_valid():
            form.instance.save()
            return redirect('open_converter')
        else:
            return self.render_to_response({'form': form})

def photo_list(request):
    photos = Photo.objects.all()
    return render(request, 'photo/list.html', {'photos': photos})
class PhotoDeleteView(DeleteView):
    model = Photo
    success_url = '/'
    template_name = 'photo/delete.html'
class PhotoUpdateView(UpdateView):
    model = Photo
    fields = ['photo']
    template_name = 'photo/update.html'

#cost funsion definition-----------------------------------------------------------------------------#

# GRADED FUNCTION: compute_content_cost
def compute_content_cost(a_C, a_G):

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [-1]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [-1]))

    # compute the cost with tensorflow (≈1 line)
    J_content = tf.reduce_sum((a_C_unrolled - a_G_unrolled)**2) / (4 * n_H * n_W * n_C)

    return J_content

# GRADED FUNCTION: gram_matrix
def gram_matrix(A):

    GA = tf.matmul(A, tf.transpose(A)) # '*' is elementwise mul in numpy

    return GA

# GRADED FUNCTION: compute_layer_style_cost
def compute_layer_style_cost(a_S, a_G):

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_H*n_W, n_C) (≈2 lines)
    a_S = tf.reshape(a_S, [n_H*n_W, n_C])
    a_G = tf.reshape(a_G, [n_H*n_W, n_C])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(tf.transpose(a_S)) #notice that the input of gram_matrix is A: matrix of shape (n_C, n_H*n_W)
    GG = gram_matrix(tf.transpose(a_G))

    # Computing the loss (≈1 line)
    J_style_layer = tf.reduce_sum((GS - GG)**2) / (4 * n_C**2 * (n_W * n_H)**2)

    return J_style_layer

def compute_style_cost(sess, model, STYLE_LAYERS):

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

# GRADED FUNCTION: total_cost
def total_cost(J_content, J_style, alpha = 10, beta = 40):

    J = alpha * J_content + beta * J_style

    return J

#img processing------------------------------------------------------------------------------#

def generate_rgb_list():
    rgb_colors_list = [
        [8, 153, 85], #1 dark green #089955
        [22, 15, 219], #2 dark blue #160fdb
        [166, 0, 255], #3 purple #a600ff
        [137, 211, 250], #4 light blue #89d3fa
        [255, 59, 121], #5 hot pink #ff3b7c
        [255, 186, 210], #6 light pink #ffbad2
        [188, 255, 181], #7 yellow green #bcffb5
        [240, 10, 10], #8 red #f00a0a
        [150, 144, 144], #9 grey #969090
        [255, 251, 28], #10 yellow #fffb1c
        [255, 221, 31], #11 deep yellow #ffdd1f
        [255, 108, 3], #12 orange #ff6403
        [79, 39, 17], #13 brown #4f2711
        [255, 207, 158], #14 peach #ffcf9e
        [0,0,0], #15 black #000000
        [255, 255, 255] #16 white #ffffff
    ]

    return rgb_colors_list

class ImgProcessor:

    @staticmethod
    def my_size(file_name):
        im = Image.open(file_name)
        width , height = im.size

        return width, height

    @staticmethod
    def im_resize(file_name, save_name, size):
        im = Image.open(file_name)
        im = im.resize(size)
        im = im.convert('RGB')
        im.save(save_name)

    def move_color(self, file_name, save_name, rgb_colors_list):
        im = Image.open(file_name)
        (width, height) = im.size
        for i in range(0, width):
            for j in range(0, height):
                im.putpixel((i, j), tuple(self.find_closest_color(im.getpixel((i, j)), rgb_colors_list)))
        im.save(save_name)

    @staticmethod
    def find_closest_color(cmp, rgb_colors_list):
        index_list = np.zeros(len(rgb_colors_list))
        for i in range(len(index_list)):
            index_list[i] = math.sqrt(math.pow(cmp[0] - rgb_colors_list[i][0], 2)
                                      + math.pow(cmp[1] - rgb_colors_list[i][1], 2)
                                      + math.pow(cmp[2] - rgb_colors_list[i][2], 2))
        index = np.argmin(index_list)
        return rgb_colors_list[index]

#summary function------------------------------------------------------------------------------#
def removeFile(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return True

def prepare_data(face, style):#모델에 넣기 위해 지정 사이즈로 변환

    ip = ImgProcessor()
    nu = Nu.CONFIG()
    bf_size = ImgProcessor.my_size(face) #원본 이미지 사이즈를 미리 저장. ip쓰면 오류남.

    if bf_size[0]<bf_size[1]: #원본 이미지 사이즈에 따라 input 사이즈 조정
        size=(300,400)
    elif bf_size[0]>bf_size[1]:
        size=(400,300)
    else:
        size=(400,400)
    nu.set_size(size[0],size[1])
    #nu.print_size()
    ip.im_resize(face,m_u+"/content_img.jpg",size)
    ip.im_resize(style,m_u+"/style_img.jpg",size)

def save_data(output, face):

    ip = ImgProcessor()
    af_size=ImgProcessor.my_size(face)
    ip.im_resize(output,m_u+"/final_image.jpg",af_size)

def start_model(sess):#모델을 돌려서 generated image 하나만 생성.

    content_image = imageio.imread(m_u+"/content_img.jpg")
    content_image = reshape_and_normalize_image(content_image)

    style_image = imageio.imread(m_u+"/style_img.jpg")
    style_image = reshape_and_normalize_image(style_image)

    generated_image = generate_noise_image(content_image)

    model = load_vgg_model("imagenet-vgg-verydeep-19.mat")

    # Assign the content image to be the input of the VGG model.
    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    out = model['conv4_2']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)

    # Assign the input of the model to be the "style" image
    sess.run(model['input'].assign(style_image))

    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]

    # Compute the style cost
    J_style = compute_style_cost(sess,model, STYLE_LAYERS)

    J = total_cost(J_content, J_style,  alpha = 10, beta = 40)

    # define optimizer (1 line)
    optimizer = tf.train.AdamOptimizer(2.0)

    # define train_step (1 line)
    train_step = optimizer.minimize(J)

    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())

    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(generated_image))

    for i in range(20):

        # Run the session on the train_step to minimize the total cost
        _ = sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

    # save last generated image
    save_image(m_u+'/generated_image.jpg', generated_image)

#------------------------------------------------------------------------------#
def converter(face, style):
    #prepare image data
    prepare_data(face, style)
    # Reset the graph
    tf.reset_default_graph()
    # Start interactive session
    sess = tf.InteractiveSession()
    start_model(sess)
    # Close sess
    tf.InteractiveSession.close(sess)
    return True

def open_converter(request):
    input=Photo.objects.filter(label=1)
    input2=str(input[0].photo)
    path="/media/"+input2

    datas={"input":path, "output":"img/i.jpg"} #img/i.jpg는 일부러 오류 내려고 넣음

    if request.method=='POST':
        num=request.POST['num']
        if num =='1':
            if (converter(m_u+"/"+input2,s_u+"/codepen/img/style1.png")):
                datas={"input":path, "output":"/media/"+"generated_image.jpg"}
        elif num =='2':
            datas={"input":path, "output":"/static/codepen/img/style2.jpg"}
        elif num =='3':
            datas={"input":path, "output":"/static/codepen/img/Style3.jpg"}
        elif num =='4':
            datas={"input":path, "output":"/static/codepen/img/style4.jpeg"}
        elif num =='5':
            datas={"input":path, "output":"/static/codepen/img/style5.jpg"}
        else :
            datas={"input":path, "output":"/static/codepen/img/style6.png"}

    return render(request,'image_upload/converter2.html',datas)
    
def change_color(request):
    return render(request,'test.html')

def get_data(request):
    switch = Photo.objects.filter(label=1)
    converted = switch[0].converted
    if converted:
        return redirect("/media/data.txt")
    else:
        return redirect("/media/nodata.txt")
