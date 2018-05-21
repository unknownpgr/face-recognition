import os
import cv2
import dlib
import time
import numpy as np
import pickle
import shutil
import openface


class FaceDetector:
    def __init__(self, torch_net_model=None, img_dim=96, use_cuda=False, verbose=True):
        self.data = []
        self.index = 0
        self.label = []
        self.label_dict = {}
        self.verbose = verbose
        self.img_dim = img_dim

        self.log("Create detector...")
        start = time.time()

        if torch_net_model is None:
            model = openface.TorchNeuralNet.defaultModel
        else:
            model = torch_net_model

        if not os.path.exists(model):
            raise Exception('Model does not exist.(' + model + ')')

        self.log("Set torch net :", model)
        self.torch_net = openface.TorchNeuralNet(model=model, imgDim=img_dim, cuda=use_cuda)
        self.torch_params = [model, img_dim, use_cuda]

        svm = cv2.ml.SVM_create()
        svm.setC(12.5)
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setGamma(0.5)
        svm.setKernel(cv2.ml.SVM_RBF)

        self.svm = svm
        self.log("Detector created in {0:.3f}".format(time.time() - start), "sec\n")

    def set_svm(self, svm):
        self.svm = svm

    def append_dir(self, label, dir):
        start = time.time()
        self.log("Append directory :", dir)
        self.label_dict[self.index] = label

        count = 0

        for file_name in os.listdir(dir):
            file_path = os.path.join(dir, file_name)
            feature = self.torch_net.forwardPath(file_path)
            self.data.append(feature)
            self.label.append(self.index)
            count += 1

        self.log(count, "data appended")
        self.log("Label :", label)
        self.log("Index :", self.index)
        self.log("Data appending finished in {0:.3f}".format(time.time() - start), "sec\n")
        self.log("")
        self.index += 1

    def append_data(self, label, datas):
        self.log("Append data")
        start = time.time()
        self.label_dict[self.index] = label

        for data in datas:
            self.label.append(self.index)
            self.data.append(data)

        self.log(len(datas), "data appended")
        self.log("Label :", label)
        self.log("Index :", self.index)
        self.log("Data appending finished in {0:.3f}".format(time.time() - start), "sec\n")
        self.log("")
        self.index += 1

    def train_model(self):
        self.log("Start training...")
        start = time.time()

        assert len(self.data) is len(self.label)

        train_data = np.array(self.data, dtype=np.float32)
        train_label = np.array(self.label, dtype=np.int32)
        self.svm.train(train_data, cv2.ml.ROW_SAMPLE, train_label)
        self.log("Training finished in {0:.3f} second.\n".format(time.time() - start))

    def predict(self, img):
        assert img is not None
        feature = np.array([self.torch_net.forward(img)], dtype=np.float32)
        index = self.svm.predict(feature)[1][0][0]
        return self.label_dict[index]

    def save(self, dir):
        self.log("Saving...")
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)
        self.svm.save(os.path.join(dir, 'svm'))
        svm_temp = self.svm
        torch_temp = self.torch_net
        self.svm = None
        self.torch_net = None
        with open(os.path.join(dir, 'detector'), 'wb') as file:
            pickle.dump(self, file)
        self.svm = svm_temp
        self.torch_net = torch_temp
        self.log("Saved.\n")

    @staticmethod
    def load(dir):
        assert os.path.exists(dir)
        start = time.time()
        with open(os.path.join(dir, 'detector'), 'rb') as file:
            detector = pickle.load(file)
            detector.svm = cv2.ml.SVM_load(os.path.join(dir, 'svm'))

            torch_params = detector.torch_params
            detector.torch_net = openface.TorchNeuralNet(torch_params[0], torch_params[1], torch_params[2])
            detector.log('Detecter loading finished in {0:.3f}'.format(time.time() - start), "sec")
            return detector

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.torch_net.__exit__(None, None, None)

    def log(self, *log):
        if self.verbose:
            for text in log:
                print(text, end=' ')
            print()
