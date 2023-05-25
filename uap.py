import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import copy
import librosa.display
from torchmetrics.functional.audio import signal_noise_ratio
from Sincnet.Sincnet import Sincnet_batch, get_name_by_label
from OneDCNN.models.OneDCNN import OneDCNN
from xvector.models.x_vector import X_vector
from myutils import WaveformDataset, Mydataset
from torch.utils.data import DataLoader


import warnings,os
warnings.filterwarnings("ignore")
gpu_id = "1,2,0"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device_ids=range(torch.cuda.device_count())

def deepfool(image, net, label, num_classes=10, overshoot=0.02, max_iter=50):
    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        image = image.cuda()
        # net = net.cuda()
    
    I = np.arange(num_classes)
    residual_labels = [l for l in I if l!=label]

    input_shape = image.shape
    pert_image = image.clone()
    w = torch.zeros(input_shape).cuda()
    r_tot = torch.zeros(input_shape).cuda()

    loop_i = 0

    # x = Variable(pert_image[None, :], requires_grad=True)
    x = torch.tensor(pert_image, requires_grad=True)
    fs = net(spectro_gram(x).unsqueeze(0))[0]
    k_i = torch.argmax(fs)
    fs = fs.requires_grad_(True)
    while k_i == label and loop_i < max_iter:
        pert = torch.inf
        # x.register_hook(print)
        fs[label].backward(retain_graph=True)
        grad_orig = x.grad.clone()

        for k in residual_labels:
            x.grad.data.zero_()

            fs[k].backward(retain_graph=True)
            cur_grad = x.grad
            
            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[k] - fs[label])#.cpu().numpy()

            pert_k = torch.abs(f_k)/torch.linalg.norm(w_k.flatten())
            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = torch.tensor(w_k)

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / torch.linalg.norm(w)
        r_tot = torch.tensor(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net(spectro_gram(x).unsqueeze(0))[0]
        k_i = torch.argmax(fs)
        # k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot
    return r_tot, loop_i, k_i

class MelSpectrogram_gpu(nn.Module):
    def __init__(self, win_length=1024, hop_length=256, n_fft=1024, n_mels=24):
        super().__init__()
        self.window = torch.hann_window(win_length).cuda()
        self.mel_basis = librosa.filters.mel(
            sr=16000,
            n_fft=n_fft,
            n_mels=n_mels
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).cuda().to(torch.float32)
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
    def forward(self, x):
        stft = torch.stft(x,
                          win_length=self.win_length,
                          hop_length=self.hop_length,
                          n_fft=self.n_fft,
                          window=self.window)
        # real = stft[:, :, :, 0]
        # im = stft[:, :, :, 1]
        # spec = torch.sqrt(torch.pow(real, 2) + torch.pow(im, 2))
        spec = torch.norm(stft, p=2, dim=-1).to(torch.float32)
        # convert linear spec to mel
        mel = torch.matmul(self.mel_basis,spec)

        # mfcc = scipy.fftpack.dct(mel, axis=0, type=4, norm="ortho")[:20]
        return mel 

def power_to_db(x):
    return 20 * torch.log10(x + 1e-5)

def normalize(inputs):
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    inputs = (inputs - inputs_m) / inputs_s 
    return inputs

def spectro_gram(waveform):
    MelSpectrogram = MelSpectrogram_gpu()
    logmelspec = MelSpectrogram(waveform)
    logmelspec = power_to_db(logmelspec)
    logmelspec = normalize(logmelspec)
    return logmelspec

# Project on the l_p ball centered at 0 and of radius 'norm_bound'
def projection_operator(v, norm_bound, p):
    if p == 2:
        v = v * min(1, norm_bound/torch.linalg.norm(v.flatten()))
    elif p == torch.inf:
        v = torch.clip(v, -norm_bound, norm_bound)
    else:
         raise ValueError("Only p = 2 or p = torch.inf allowed")
    return v

class Attack:
    def __init__(self):
        self.TIMIT_dataset = '/mnt/hhddata2t1/yejianbin/TIMIT_dataset/TIMIT/test'
        self.dataset = '/mnt/hhddata2t1/yejianbin/AudioMNIST.npz'

        arch = OneDCNN()
        checkpoint = 'checkpoint/BadNet1.pth'
        self.classifier = self.load_model(arch, checkpoint)

    def load_model(self, model, checkpoint):
        for param in model.parameters():
            param.requires_grad = False
        model.cuda()
        if len(device_ids)>1:
            model = torch.nn.DataParallel(model)
        checkpoint = torch.load(checkpoint)
        # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint["backdoor"].items()})
        model.load_state_dict(checkpoint["backdoor"])
        model.eval()
        return model

    def build_dataset(self):
        # test_dataset = WaveformDataset(input_path=self.TIMIT_dataset)
        # self.dataset_size = len(test_dataset)
        # print(f'Dataset size: {self.dataset_size}')
        # # test_loader = AudioDataLoader(dataset=test_dataset, batch_size=1)
        # return test_dataset

        path = self.dataset
        f = np.load(path)
        x, y = f['x'], f['y']
        f.close()
        # if size:
        #     x, y = x[:size], y[:size]
        dataset = Mydataset(x, y)
        dataset_size = len(dataset)
        self.train_size = int(0.8 * dataset_size)
        self.validate_size = dataset_size - self.train_size
        train_set, validate_set = random_split(dataset, [self.train_size, self.validate_size])
        print(f'Dataset size: {dataset_size}')
        return train_set, validate_set

    def add_perturbation(self, audio, v):
        data_len = audio.shape[0]
        perturbation_len = len(v)
        tmp_pert = v.clone()
        if data_len > perturbation_len:
            n = int(data_len / perturbation_len)
            tmp_pert = v.repeat(1,n+1)[0]    # repeat
        if data_len < len(tmp_pert):
            tmp_pert = tmp_pert[:data_len]  #   clip
        audio = audio + tmp_pert

        return audio

    def universal_perturbation(self, alpha=0.1,
                                max_iter_uni=5,
                                num_classes=10,
                                norm_bound=0.1,
                                p=2,
                                max_iter_df=100):
        fooling_rate = 0.0
        prev_iter_fooling_rate = 0.0
        itr = 0
        train_set, validate_set = self.build_dataset()

        #Fooling ratio of the perturbation in both training and validation sets
        global_fooling_rate = 0.0
        global_fooling_rate_valid = 0.0

        UAP = torch.zeros((16000),requires_grad=True).cuda()

        #Compute the estimated labels for the original training dataset
        max_magnitude = 0
        est_labels_orig = torch.zeros((self.train_size))
        true_ground = torch.zeros((self.train_size))
        for ii in range(self.train_size):
            data, label = train_set[ii]
            # true_ground[ii] = label
            if np.max(np.abs(data)) > max_magnitude:
                max_magnitude = np.max(np.abs(data))
            # data = torch.tensor(data).cuda()
            # logmelspec = spectro_gram(data).unsqueeze(0)
            # labels_orig = self.classifier(logmelspec)
            # est_labels_orig[ii] = torch.argmax(labels_orig)
            est_labels_orig[ii] = label

        #Compute also the labels for the original validation dataset
        est_labels_orig_valid = torch.zeros((self.validate_size))
        for ii in range(self.validate_size):
            _, label = validate_set[ii]
            est_labels_orig_valid[ii] = label
        
        while (fooling_rate < 1-alpha) and (itr < max_iter_uni):
            for k in range(0, self.train_size):
                cur_sample, _ = train_set[k]
                cur_sample = torch.tensor(cur_sample).cuda()
                label = est_labels_orig[k]

                scores = self.classifier(spectro_gram(self.add_perturbation(cur_sample, UAP)).unsqueeze(0))
                cur_label = torch.argmax(scores)
                if label == cur_label:
                    print('>> k = ', k, ', pass #', itr)
                    print("Global fooling rate:", str(global_fooling_rate))
                    dr, iter, adv_pred = deepfool(self.add_perturbation(cur_sample, UAP), 
                                                self.classifier, int(label.item()), 
                                                num_classes=num_classes, 
                                                overshoot=0.1, 
                                                max_iter=max_iter_df)
                    # Check the fooling rato after the evaluation
                    est_labels_pert = torch.zeros((self.train_size))
                    FOOLING_RATIO_NOT_IMPROVABLE = False
                    
                    train_loader = DataLoader(dataset = train_set, batch_size=128, shuffle=False, num_workers=6)
                    num=0
                    for ii, (data) in enumerate(train_loader):
                        original_sample_batch = data[0].cuda()
                        cur_batch_size = original_sample_batch.shape[0]
                        perturbed = original_sample_batch + projection_operator(UAP + dr, norm_bound, p)
                        perturbed = spectro_gram(perturbed)
                        logits_pert = self.classifier(perturbed)
                        est_labels_pert[num : num+cur_batch_size] = torch.argmax(logits_pert,axis=1)
                        num += cur_batch_size
                        
                        #Tip for computational efficiency:
                        #If we already know that we can not improve the global fooling rate, stop computing the fooling ratio
                        pending_samples = self.train_size - num
                        max_reachable_fooling_rate =  float( (torch.sum(est_labels_pert[0:num]!=est_labels_orig[0:num])+pending_samples)/float(self.train_size) )
                        if max_reachable_fooling_rate < global_fooling_rate:
                            FOOLING_RATIO_NOT_IMPROVABLE = True
                            print("%d) F.R. NOT IMPROVABLE: MAX. %d+%d SAMPLES CORRECT OUT OF %d"%(num, int(torch.sum(est_labels_pert[0:num]!=est_labels_orig[0:num])) , pending_samples , self.train_size))
                            break

                    #If the fooling ratio can not be improved, set a small value for the current fooling ratio, just to reject updating the perturbation
                    if FOOLING_RATIO_NOT_IMPROVABLE:
                        local_fooling_rate = 0.0 #
                    else:
                        local_fooling_rate = float(torch.sum(est_labels_pert[0:num] != est_labels_orig[0:num]) / float(self.train_size))


                    # Make sure it converged...
                    if iter < max_iter_df-1 and local_fooling_rate>global_fooling_rate:
                        UAP = UAP + dr
                        UAP = projection_operator(UAP, norm_bound, p) # Project on l_p ball
                        
                        #UPDATE THE GLOBAL FOOLING RATE
                        global_fooling_rate = local_fooling_rate

                    else:
                        print("Not converged or global fooling rate not improved")

            db_diff_max = signal_noise_ratio(cur_sample, UAP)
            print(f'PSNR: {db_diff_max}')

            # Perturb the dataset and compute the fooling rate on the training set
            train_loader = DataLoader(dataset = train_set, batch_size=128, shuffle=False, num_workers=6)
            est_labels_pert = torch.zeros((self.train_size))
            num=0
            for ii, (data) in enumerate(train_loader):
                original_sample_batch = data[0].cuda()
                cur_batch_size = original_sample_batch.shape[0]
                perturbed = original_sample_batch + UAP
                perturbed = spectro_gram(perturbed)
                logits_pert = self.classifier(perturbed)
                est_labels_pert[num : num+cur_batch_size] = torch.argmax(logits_pert,axis=1)
                num+= cur_batch_size
            
            # Compute the fooling rate
            fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(self.train_size))
            print('FOOLING RATE = ', fooling_rate)
            # print(est_labels_pert)
            fooling_rates_vec[itr] = fooling_rate


            # Perturb the dataset and compute the fooling rate on the validation set
            validate_loader = DataLoader(dataset = validate_set, batch_size=128, shuffle=False, num_workers=6)
            est_labels_pert_valid = torch.zeros((self.validate_size))
            num=0
            for ii, (data) in enumerate(validate_loader):
                original_sample_batch = data[0].cuda()
                cur_batch_size = original_sample_batch.shape[0]
                perturbed = original_sample_batch + UAP
                perturbed = spectro_gram(perturbed)
                logits_pert = self.classifier(perturbed)
                est_labels_pert_valid[num : num+cur_batch_size] = torch.argmax(logits_pert,axis=1)
                num+= cur_batch_size

            # Compute the fooling rate
            fooling_rate_valid = float(np.sum(est_labels_pert_valid != est_labels_orig_valid) / float(self.validate_size))
            print('FOOLING RATE (VALID) = ', fooling_rate_valid)
            # print(est_labels_pert_valid)
            fooling_rates_vec_valid[itr] = fooling_rate_valid


            #If the fooling ratio has not changed in the whole epoch, finish the algorithm
            if np.abs(prev_iter_fooling_rate-fooling_rate)<0.000000001:
                print("Finishing at iteration " + str(itr) + " --> FR has not changed in the whole epoch")
                break

            #Store the current fooling rate, to use it in the next epoch
            prev_iter_fooling_rate = fooling_rate

            itr = itr + 1

if __name__ == '__main__':
    attack = Attack()
    attack.universal_perturbation()