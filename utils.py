from myImports import *


class Utils:

    #Preprocessing image
    def pre_processing(self, img):
        x, y = img.size
        size = max(512, x, x)
        new_im = Image.new('RGB', (size, size), (0, 0, 0))
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

        # Intermediate Function to process data from the data retrival class
    def prepare_data(self, DF):
        from CBIRDataset import CBIRDataset
        trainDF, validateDF = train_test_split(DF,
                                               test_size=0.15,
                                               random_state=RANDOMSTATE)
        train_set = CBIRDataset(trainDF)
        validate_set = CBIRDataset(validateDF)

        return train_set, validate_set

    def getImageClass(self, imagePath):
        txt = imagePath
        x = txt.split("\\")
        classeIdx = len(x) - 2
        return (x[classeIdx])

    def getImageID(self, filename):
        txt = filename
        x = txt.split(".")
        return (x[0])

    def getTransformations(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Load Model in Evaluation phase
    def load_model_from_dir(self, modelPath):
        from ConvAutoencoder_v2 import ConvAutoencoder_v2
        model = ConvAutoencoder_v2().to(device)
        model.load_state_dict(torch.load(
            modelPath, map_location=device)['model_state_dict'],
                              strict=False)
        #model.eval()
        return model

    def get_image_latent_features(self, img_path, transformations, model):

        img = Image.open(img_path)  #.convert('RGB')
        new_img = self.pre_processing(img)
        tensor = transformations(new_img).to(device)
        # tensor = tensor.reshape(1, 1, 512, 303)
        latent_features = model.encoder(
            tensor.unsqueeze(0)).cpu().detach().numpy()
        return latent_features

    def get_latent_features(self, images, transformations, model):

        latent_features = np.zeros((len(images), 256, 16, 16))

        for i, image in enumerate(tqdm(images)):
            img = Image.open(image)  #.convert('RGB')
            new_img = self.pre_processing(img)
            tensor = transformations(new_img).to(device)
            # tensor = tensor.reshape(1, 1, 512, 303)
            latent_features[i] = model.encoder(
                tensor.unsqueeze(0)).cpu().detach().numpy()

        del tensor
        gc.collect()
        return latent_features

    def euclidean(self, a, b):
        # compute and return the euclidean distance between two vectors
        return np.linalg.norm(a - b)

    def perform_search(self, queryFeatures, index, maxResults):

        results = []

        for i in range(0, len(index["features"])):
            # compute the euclidean distance between our query features
            # and the features for the current image in our index, then
            # update our results list with a 2-tuple consisting of the
            # computed distance and the index of the image
            d = self.euclidean(queryFeatures, index["features"][i])
            results.append((d, i))

        # sort the results and grab the top ones
        results = sorted(results)[:maxResults]
        # return the list of results
        return results

    def features_extraction_and_index_creation(self, df, transformations,
                                               model):
        #extract features and store them in index
        imagesPaths = df.image.values
        classes = df.classe.values
        filesNames = df.filename.values
        diractoriesNames = df.directoryname.values
        latent_features = self.get_latent_features(imagesPaths,
                                                   transformations, model)

        indexes = list(range(0, len(imagesPaths)))
        feature_dict = dict(zip(indexes, latent_features))
        index_dict = {'indexes': indexes, 'features': latent_features}

        return imagesPaths, classes, latent_features, index_dict, filesNames, diractoriesNames

    #bloody mAP
    def averagePrecision(self, maxResults, index_dict, latent_features,
                         imagesLength):
        mvp = 0
        for queryIndex in range(0, len(index_dict["indexes"])):
            queryFeatures = latent_features[queryIndex]
            results = self.perform_search(queryFeatures, index_dict,
                                          maxResults)
            precisions = self.getPrecisions(maxResults, results)
            binaryLable = self.getImagesBinaryLabled(results, queryIndex)
            ap = average_precision_score(binaryLable, precisions)
            print(queryIndex, ' =========== ', ap)
            mvp = mvp + ap
        return mvp / imagesLength  #len(images)

        # averagePrecision(100)

    def precV2(self, qIdx, rslt, classes):
        #Query image classe
        QueryImageClasse = classes[qIdx]

        # getNbrOfTotalImagesInClass
        ttImagesInClass = 0
        for i in classes:
            if classes[qIdx] == i:
                ttImagesInClass = ttImagesInClass + 1
            #print('totalImagesInClass',ttImagesInClass)
        #getCorrectRetreivedimage
        correctImages = []
        for x in rslt:
            # print('i',i)
            #print(classes[x[1]])
            if classes[qIdx] == classes[x[1]]:
                correctImages.append(classes[x[1]])
        #print('Correct images retreived = ', len(correctImages), maxRetreivedImages)
        return len(correctImages) / len(rslt)
        # precV2(queryIdx, results)

    def getPrecisions(self, maxRetreivedImages, results, classes):
        pre_tab = []
        for ikp in range(maxRetreivedImages):
            #print('ikp = ' , ikp + 1)
            for (d, j) in results[:ikp + 1]:
                #print('jk = ' , j)
                prec = self.precV2(j, results[:ikp + 1], classes)
            #print(prec , ' in ',  len(results[:ikp+1]), '    classe = ', classes[j])
            pre_tab.append(prec)
        return pre_tab
        # pre_tab = getPrecisions(maxRetreivedImages,results)

    def getImagesBinaryLabled(self, results, queryIdx, classes):
        binaryLabels = []
        for (d, j) in results:
            if (classes[j] == classes[queryIdx]):
                binaryLabels.append(1)
            elif (classes[j] != classes[queryIdx]):
                binaryLabels.append(0)
        return binaryLabels

    def load_ckpt(self, checkpoint_fpath, model, optimizer):

        # load check point
        checkpoint = torch.load(checkpoint_fpath)

        # initialize state_dict from checkpoint to model
        model.load_state_dict(checkpoint['model_state_dict'])

        # initialize optimizer from checkpoint to optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # initialize valid_loss_min from checkpoint to valid_loss_min
        #valid_loss_min = checkpoint['valid_loss_min']

        # return model, optimizer, epoch value, min validation loss
        return model, optimizer, checkpoint['epoch']

    def save_checkpoint(self, state, filename):
        """Save checkpoint if a new best is achieved"""
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint

    def train_model(
            self,
            model,
            dataloaders,
            dataset_sizes,
            criterion,
            optimizer,
            scheduler,  #this was commented 
            num_epochs):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = np.inf

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                for idx, inputs in enumerate(Bar(dataloaders[phase])):
                    inputs = inputs.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, inputs)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                            # m = nn.Sigmoid()
                            # loss = nn.BCELoss()
                            # input = torch.randn(3, requires_grad=True)
                            # target = torch.empty(3).random_(2)
                            # output = loss(m(input), target)
                            # output.backward()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                if phase == 'train':  #this was commented
                    scheduler.step()  #this was commented

                epoch_loss = running_loss / dataset_sizes[phase]

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    self.save_checkpoint(
                        state={
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'best_loss': best_loss,
                            'optimizer_state_dict': optimizer.state_dict()
                        },
                        filename='ckpt_epoch_{}.pt'.format(epoch))

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, optimizer, epoch_loss


# from google.colab import files
# src = list(files.upload().values())[0]
# open('mymodel.py','wb').write(src)
# import mymodel