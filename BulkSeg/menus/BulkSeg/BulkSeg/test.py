import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

import numpy as np
from data_loader import cellpose
from skimage import io

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)


    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    # state_dict = checkpoint
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))


    print("length = ", len(data_loader))

    with torch.no_grad():
        for i, (data, (original_heights, original_widths)) in enumerate(tqdm(data_loader)):
            data = data.to(device)

            output = model(data)

            for j, (flow_prob, original_height, original_width) in enumerate(zip(output.cpu().numpy(), original_heights.numpy(), original_widths.numpy())):
                flow_prob = cellpose.util.resize(flow_prob, (original_height, original_width))
                flow_prob[2] = cellpose.util.sigmoid_func(flow_prob[2])
                flow_prob = np.transpose(flow_prob, (1,2,0))

                np.save(str(i)+"-"+str(j)+"-"+"flow-after-training.npy", flow_prob)

                lab = cellpose.util.flow2msk(flow_prob)

                lab = lab[:, :, None]
                io.imsave(str(i)+"-"+str(j)+"-lab.png", lab.astype(np.uint8))

                # #### computing loss, metrics on test set
                # loss = loss_fn(output, target)
                # batch_size = data.shape[0]
                # total_loss += loss.item() * batch_size
                # for i, metric in enumerate(metric_fns):
                #     total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
