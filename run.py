import os,inspect,warnings,argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
warnings.filterwarnings('ignore')

import tensorflow as tf
import model.dataset as dman
import model.network as nn
import model.train as tr
import model.test as te

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
CKPT_DIR = PACK_PATH+'/Checkpoint'

def main():
    
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    
    dataset = dman.Dataset(normalize=FLAGS.datnorm)
    neuralnet = nn.CNN(height = dataset.height, width=dataset.width, channel=dataset.channels,
                      num_classes = dataset.num_classes, ksize=3, learning_rate=FLAGS.lr, chkpt_dir=CKPT_DIR)
    
    tr.training(neuralnet=neuralnet, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, normalize=True)
    te.test(neuralnet=neuralnet, dataset=dataset, batch_size=FLAGS.batch)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--lr', type=int, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=15, help='Training epoch')
    parser.add_argument('--batch', type=int, default=128, help='Mini batch size')
    FLAGS, unparsed = parser.parse_known_args()

    main()
    