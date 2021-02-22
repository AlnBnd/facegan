import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import os
import io
import IPython.display  
import random 

def read_feature(file_name):
    file = open(file_name, mode='r')
    contents = file.readlines()
    code = np.zeros((512, ))
    for i in range(512):
        name = contents[i]
        name = name.strip('\n')
        code[i] = name
    code = np.float32(code)
    file.close()
    return code

def move_latent_and_save(latent_vector, size_img, direction_intensity, Gs_network, Gs_syn_kwargs):
    os.makedirs('results/param', exist_ok=True)

    new_latent_vector = latent_vector.copy()
    new_latent_vector[0][:8] = (latent_vector[0] + direction_intensity)[:8]
    images = Gs_network.components.synthesis.run(new_latent_vector, **Gs_syn_kwargs)
    result = PIL.Image.fromarray(images[0], 'RGB')
    result.thumbnail(size_img, PIL.Image.ANTIALIAS)
    data = random.randint(0,100)
    result.save('results/param/'+str(data)+'.png')
    # io_data = io.BytesIO()
    # im_data = io_data.getvalue()
    # disp = IPython.display.display(IPython.display.Image(im_data))
    return result


def read_face_latent():
    tflib.init_tf()
    with open('networks/stylegan2-ffhq-config-f.pkl', "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    w_avg = Gs_network.get_var('dlatent_avg')
    noise_vars = [var for name, var in Gs_network.components.synthesis.vars.items() if name.startswith('noise')]
    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1
    truncation_psi = 0.5
  

    face_latent = read_feature('results/generate_codes/0000.txt')
    z = np.stack(face_latent for _ in range(1))
    tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
    w = Gs_network.components.mapping.run(z, None)
    w = w_avg + (w - w_avg) * truncation_psi
    return w, Gs_network, Gs_syn_kwargs

