import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


STYLE_WEIGHT = 1
CONTENT_WEIGHT = 10000000

EPOCHS = 1
STEPS_PER_EPOCH = 100

def load_img(path_to_img,max_dim=512):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def create_mini_model(model,style_layers,content_layers): 
    outputs = [model.get_layer(name).output for name in style_layers + content_layers]
    inputs = model.input

    model = tf.keras.models.Model([vgg.input],outputs)
    return model

def gram_matrix(tensor):
    gram_matrix = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    return gram_matrix/(tensor.shape[1]*tensor.shape[2])

def calculate_loss(outputs):
    style_outputs = outputs['style']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight/len(style_outputs)

    content_outputs = outputs['content']
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                           for name in content_outputs.keys()])
    content_loss *= content_weight/len(content_outputs)
    
    total_loss = style_loss + content_loss
    return total_loss

def train_step(image):
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = calculate_loss(outputs)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

class StyleContent_Extractor(tf.keras.models.Model):
    def __init__(self,model,style_layers, content_layers):
        super(StyleContent_Extractor, self).__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.model = create_mini_model(model,style_layers,content_layers)
        self.model.trainable = False

          
    def call(self, inputs):
        inputs = inputs*255.0
        preprocessed_inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.model(preprocessed_inputs)

        style_outputs = outputs[:len(self.style_layers)]
        content_outputs = outputs[len(self.style_layers):]

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {
              content_name:output 
              for content_name,output in zip(self.content_layers,content_outputs)
        }

        style_dict = {
              style_name:output 
              for style_name,output in zip(self.style_layers,style_outputs)
        }

        return {'style':style_dict,'content':content_dict}


try:
    content_path = tf.keras.utils.get_file('content_img', sys.argv[2])
except:
    content_path = tf.keras.utils.get_file('content_img', 'https://i.imgur.com/F28w3Ac.jpg')

try:
    style_path = tf.keras.utils.get_file('style_img', sys.argv[3])
except:
    style_path = tf.keras.utils.get_file('styleimg','https://i.imgur.com/UWIRzW9.jpeg')


content_img = load_img(content_path)
style_img = load_img(style_path)

vgg = tf.keras.applications.VGG19(include_top=False)
vgg.save("./.vgg")
vgg.training = False


content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 

                'block5_conv1']

extractor = StyleContent_Extractor(vgg,style_layers, content_layers)
style_targets = extractor(style_img)['style']
content_targets = extractor(content_img)['content']

#scaling weights for faster calculations
style_weight = STYLE_WEIGHT / 1e5
content_weight = CONTENT_WEIGHT / 1e5

generated_img = tf.Variable(content_img)

step = 0
for n in range(EPOCHS):
    for m in range(STEPS_PER_EPOCH):
        step += 1
        train_step(generated_img)
        print("-+", end='')
    print(f"Train step: {step}")
plt.imsave(sys.argv[1],tf.squeeze(generated_img).numpy())