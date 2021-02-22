import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import os

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

def move_latent_and_save(latent_vector, direction_file, coeffs, Gs_network, Gs_syn_kwargs):
    direction = np.load('latent_directions/' + direction_file)
    os.makedirs('results/'+direction_file.split('.')[0], exist_ok=True)

    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[0][:8] = (latent_vector[0] + coeff*direction)[:8]
        images = Gs_network.components.synthesis.run(new_latent_vector, **Gs_syn_kwargs)
        result = PIL.Image.fromarray(images[0], 'RGB')
        result.save('results/'+direction_file.split('.')[0]+'/'+str(i).zfill(3)+'.png')


def main():

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

    direction_file = ['age.npy', 'angle_horizontal.npy', 'emotion_happy.npy', 'emotion_sad.npy', 'angle_pitch.npy',
                        'gender.npy'] 


    coeffs = [-12., -9., -6., -3., 0., 3., 6., 9., 12.]

    for i_file in direction_file:
        move_latent_and_save(w, i_file, coeffs, Gs_network, Gs_syn_kwargs)


if __name__ == "__main__":
    main()
