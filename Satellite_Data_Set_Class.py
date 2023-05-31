class Satellite_Data_Set_Class(torch.utils.data.Dataset):


  def __init__(self, root_dir, csv_file_dir=False, csv_file=False, transform=None, activation_code=None, prediction_mode='classification', sensitivity=0, name='native'):
    """
    Parameters:
      csv_file_dir (str): path to the original csv file
      csv_file (csv): alternative to csv_file_dir; used if modifications to csv_file are necessary
      root_dir (str): path to the directory containing all files
      transform (callable): optional transformer
      activation_code (False, list): list containing the indices of the required data in the original csv file. If False, take all the availablle data
                                     Important! To obtain such a list use the train_val_test_split method on the complete dataset first before instantiating a train class
      prediction_mode (str): modifies what type of labels will be called in the __getitem__() function
          'classification' : the labels are optimised for classification. The classification depends on the 'sensitivity' variable (0.5, 1)
          'multiregression': the labels are optimised for multivariate regression
    """   
    self.activation_code = activation_code 
    if activation_code:
      if isinstance(csv_file_dir, str):
        csv_content = pd.read_csv(csv_file)
        csv_content = csv_content[csv_content.index.isin(self.activation_code)]
        self.frame = csv_content
      elif isinstance(csv_file, pd.DataFrame):
        csv_content = csv_file
        csv_content = csv_content[csv_content.index.isin(self.activation_code)]
        self.frame = csv_content
      else:
        raise ValueError('The csv file has not been added correctly or is missing')
    else:
      if isinstance(csv_file_dir, str):
        self.frame = pd.read_csv(csv_file)
      elif isinstance(csv_file, pd.DataFrame):
        self.frame = csv_file
      else:
        raise ValueError('The csv file has not been added correctly or is missing')

    self.root_dir = root_dir
    self.transform = transform
    self.sensitivity = sensitivity
    self.prediction_mode = prediction_mode
    if activation_code:
      self.name = str(name)
    else:
      self.name = 'native'


  # return a list containing all keys contained in the csv file
  def get_keys(self):
    list_with_keys = list(self.frame.columns[2:])
    list_with_keys.append('mixed')
    return list_with_keys

  # alternative length function as the torch implementation may struggle
  def __len__(self):
    return len(self.frame)



  # return the loaded .npy file with labels in a dictionary
  def _get_single_item(self, idx, transform=False, data_type='float', layers=[3, 2, 1]):
    """
    This is an internal function, capable of getting only a single item at a time. This function is used in sds.get_item()

    Parameters:
      idx (int): position of required image in the csv file
      transform (callable): optional transformer
    """    

    # get the image, load it, then get the labels. Save all the data in a dic
    img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0]) 
    if data_type == 'float':
      image = np.load(img_name, allow_pickle=True).astype(np.float32, copy=False)[layers]
    if data_type == 'int':
      image = np.load(img_name, allow_pickle=True).astype(np.int32, copy=False)[layers]
    ############## TEMP SOLUTION
    """image = np.squeeze(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))"""
  
    ############## TEMP SOLUTION
    image = torch.from_numpy(image)
    if self.prediction_mode == 'multiregression':
      labels = self.frame.iloc[idx, 2:]
      labels = np.array([labels]).astype(np.float32, copy=False)
      labels = labels.astype('float').reshape(-1, 11)
      labels = torch.from_numpy(labels)[0]
    if self.prediction_mode == 'classification':
      labels = self._get_category(idx, sensitivity=self.sensitivity, translated=False)
      labels = np.array([labels])
      labels = torch.from_numpy(labels)[0]
    
    if layers == [3, 2, 1, 4]:
      if transform:
        if self.transform:
          image = self.transform(image)
        else:
          warnings.warn('There was no transformer added')

    sample = {'image': image, 'label': labels}
    return sample
    
  def __getitem__(self, idx, transform=True, data_type='float', layers=[3, 2, 1, 4]):
    """
    iterates over the input data and returns a list containing dic which contain the image/labels

    Parameters:
      idx (int, list, tensor): position of required image in the csv file
      transform (callable): optional transformer
      data_type ('float', 'int'): get the tensor containing either float or int. For Neural Networks float is recommended
    """    
    if torch.is_tensor(idx):
      idx = idx.tolist()

    if isinstance(idx, int):
      return self._get_single_item(idx, transform=transform, data_type=data_type, layers=layers)
    else:
      # This function returns a list containing dic with the data of all samples
      # the map function is used due to its performance increase over for loops (the training time is decreased by +25%)
      # the use of the lambda function is necessary due to the optional parameters in the _get_single_item() function
      return list(map(lambda iterable: self._get_single_item(iterable, transform=transform, data_type=data_type, layers=layers), idx))    

  def _get_category(self, idx, translated=False, sensitivity=0):
    """
    Internal Function, able to get the category of a single image at a time based off of its position in the csv file

    Parameters:
      idx (int): position of required images category in the csv file
      translated (bool): if True, get the category as string. Otherwise as int
      sensitiviy (float [0, or range(0.5, 1)]): the percentage of the picture which must be covered by a single category
    """    

    list_containing_all_categories = [self.frame.iloc[idx, 2:]]
    if np.max(np.array(list_containing_all_categories)) <= sensitivity:
      if translated:
        return 'mixed'
      else:
        return 11
    elif sensitivity >= 0.5:
      index_of_value_making_up_majority_of_image = np.where(np.array(list_containing_all_categories)>sensitivity)[1][0]
      if translated:
        return self.get_keys()[index_of_value_making_up_majority_of_image]
      else:
        return index_of_value_making_up_majority_of_image
    else:
      index_of_value_making_up_majority_of_image = np.where(np.array(list_containing_all_categories)==np.max(np.array(list_containing_all_categories)))[1][0]
      if translated:
        return self.get_keys()[index_of_value_making_up_majority_of_image]
      else:
        return index_of_value_making_up_majority_of_image


      

  def get_category(self, idx, translated=False, sensitivity=0):
    """
    Parameters:
      idx (int, list): position of required images category in the csv file
      translated (bool): if True, get the category as string. Otherwise as int
      sensitiviy (float [0, or range(0.5, 1)]): the percentage of the picture which must be covered by a single category
    """        
    if isinstance(idx, int):
      sample_list = self._get_category(idx, translated=translated, sensitivity=sensitivity)
    else:
      sample_list = list(map(lambda iterable: self._get_category(iterable, translated=translated, sensitivity=sensitivity), idx))
    return sample_list

  def assign_category(self, idx=False, sensitivity=0, translated=False):
    """
    Return a dictionary containing the indices of all images for a given category

    Parameters:
      idx (bool|list) : if False, return category of every img in self.frame. If list, return only for values inside list
      sensitiviy (float [0.5, 1]): the percentage of the picture which must be covered by a single category
      translated (bool): if True, get the category as string. Otherwise as int
    """  
    if idx == False:
      all_categories = self.get_category(list(range(self.__len__())), translated=translated, sensitivity=sensitivity)
      dic_split_into_categories = {}
      for category in set(self.get_category(list(range(self.__len__())), translated=translated, sensitivity=sensitivity)):
        dic_split_into_categories[category] = np.where(np.isin(all_categories, category))[0].tolist()
      return dic_split_into_categories
    else:
      all_categories = self.get_category(idx, translated=translated, sensitivity=sensitivity)
      dic_split_into_categories = {}
      for category in set(self.get_category(idx, translated=translated, sensitivity=sensitivity)):
        #Because not the complete dataset is taken in this case, np.where will not return the identical results as in the above version
        correct_idx_list = [idx[i] for i in np.where(np.isin(all_categories, category))[0].tolist()]
        dic_split_into_categories[category] = correct_idx_list
      return dic_split_into_categories
  
  def _only1(l):
    """
    Check if a list of items has a single True value. Required to ensure sound logic of some functions
    """
    return l.count(True) == 1
  
  def _display_single_image(self, idx, display=True, output=False, layer='rgb', sensitivity=0):
    """
    get a single image that may be displayed
    used in display_image()

    Parameters: 
      idx (int, list, tensor): position of required image in the csv file
      display (bool)         : if True, display a *single* image. If True, Output must be False
      output (bool)          : if True, return a tuple containing idx and the normalized image as np.array. If True, display must be False
      layer (str)            : The layer which will be displayed or returned. Only one layer may be selected at once
                        'rgb': the red, green, blue layers are combined to view the image as seen by humans
                   'infrared': highlight urban/dead organic material
             'short_infrared': shows organic matter. The darker, the denser. But brown color shows barren areas
           'vegetation_index': high numbers designate vegetation, low numbers designate urban/water areas
             'moisture_index': the lower the number, the wetter an area
    """
    # Source: https://gisgeography.com/sentinel-2-bands-combinations/

    if output and display:
      raise ValueError('Output and Display may not be True simultaneously')
    if not output and not display:
      raise ValueError('Either Output or Display must be True')
    
    if layer == 'rgb':
      image = self.__getitem__(idx, layers=[3, 2, 1])['image']
      image = np.squeeze(image.numpy())
      image = (image - np.min(image)) / (np.max(image) - np.min(image))
      image = image.transpose((1, 2, 0))

    elif layer == 'infrared':
      image = self.__getitem__(idx, layers=[7, 3, 2])['image']
      image = np.squeeze(image.numpy())
      image = (image - np.min(image)) / (np.max(image) - np.min(image))
      image = image.transpose((1, 2, 0))

    elif layer == 'short_infrared':
      image = self.__getitem__(idx, layers=[11, 7, 3])['image']
      image = np.squeeze(image.numpy())
      image = (image - np.min(image)) / (np.max(image) - np.min(image))
      image = image.transpose((1, 2, 0))

    elif layer == 'vegetation_index':
      temp_layer_7 = self.__getitem__(idx, layers=[7])['image']
      temp_layer_3 = self.__getitem__(idx, layers=[3])['image']
      final_layer = torch.div(torch.subtract(temp_layer_7, temp_layer_3), torch.add(temp_layer_7, temp_layer_3))
      image = np.squeeze(final_layer.numpy())
      image = (image - np.min(image)) / (np.max(image) - np.min(image))
      image = image.transpose((0, 1))
    
    elif layer == 'moisture_index':
      temp_layer_7 = self.__getitem__(idx, layers=[7])['image']
      temp_layer_10 = self.__getitem__(idx, layers=[10])['image']
      final_layer = torch.div(torch.subtract(temp_layer_7, temp_layer_10), torch.add(temp_layer_7, temp_layer_10))
      image = np.squeeze(final_layer.numpy())
      image = (image - np.min(image)) / (np.max(image) - np.min(image))
      image = image.transpose((0, 1))
    
    else:
      raise ValueError('This layer type has not been implemented. Please see the description for the supported layers')

    if output:
      return (idx, image)
    if display:
      title_for_image = self.get_category(idx, translated=True, sensitivity=sensitivity)
      plt.title(title_for_image)
      plt.imshow(image)
    
  def display_image(self, idx, display=False, output=True, layer='rgb'):
    """
    get multiple images as np.array, or display a single image

    Parameters: 
      idx (int, list, tensor): position of required image in the csv file
      display (bool)         : if True, display a *single* image. If True, Output must be False
      output (bool)          : if True, return a tuple containing idx and the normalized image as np.array. If True, display must be False
      layer (str)            : The layer which will be displayed or returned. Only one layer may be selected at once
                        'rgb': the red, green, blue layers are combined to view the image as seen by humans
                   'infrared': highlight urban/dead organic material
             'short_infrared': shows organic matter. The darker, the denser. But brown color shows barren areas
           'vegetation_index': high numbers designate vegetation, low numbers designate urban/water areas
             'moisture_index': the lower the number, the wetter an area
    """
    if isinstance(idx, int):
      return self._display_single_image(idx, display=display, output=output, layer=layer)
    else:
      return list(map(lambda iterable: self._display_single_image(iterable, display=display, output=output, layer=layer), idx))    

  def _get_single_feature(self, idx, display=True, output=False, feature='entropy_sum'):
    """
    get the feature of a single image

    Parameters: 
      feature (str)          : The layer which will be displayed or returned. Only one layer may be selected at once
                'entropy_sum': The entropy of the image is summed up over the whole image
               'entropy_mean': The mean of the entropy over the whole image
                'entropy_max': The entropy of the image is summed up over the whole image (different behaviour in train_val_test_split compared to entropy_sum)
                'entropy_min': The entropy of the image is summed up over the whole image (different behaviour in train_val_test_split compared to entropy_sum)
         'moisture_index_sum': The moisture_index of the image is summed up over the whole image
        'moisture_index_mean': The mean of the moisture index over the whole image
         'moisture_index_max': The moisture index of the image is summed up over the whole image (different behaviour in train_val_test_split compared to moisture_index_sum)
         'moisture_index_min': The moisture index of the image is summed up over the whole image (different behaviour in train_val_test_split compared to moisture_index_sum)
       'vegetation_index_sum': The vegetation_index of the image is summed up over the whole image
      'vegetation_index_mean': The mean of the vegetation_index over the whole image
       'vegetation_index_max': The vegetation_index of the image is summed up over the whole image (different behaviour in train_val_test_split compared to vegetation_index_sum)
       'vegetation_index_min': The vegetation_index of the image is summed up over the whole image (different behaviour in train_val_test_split compared to vegetation_index_sum)
                   'edge_sum': The edges of the image is summed up over the whole image
                  'edge_mean': The mean of the edges over the whole image
                   'edge_max': The edges of the image is summed up over the whole image (different behaviour in train_val_test_split compared to edge_sum)
                   'edge_min': The edges of the image is summed up over the whole image (different behaviour in train_val_test_split compared to edge_sum)
                  
      display (bool)         : if True, display a summary of the distribution of features
      sensitiviy (float [0.5, 1]): the percentage of the picture which must be covered by a single category. 
                                   Where no category covers more than 50%, the image will be defined as 'mixed'
      steps (int)            : The number of groups the features will be distributed on. The higher the number, the higher the risk of a group being empty
      output (bool)          : if True, return a 
     ['entropy_sum', 'moisture_index_sum', 'vegetation_index_sum', 'edge_sum', 'entropy_mean', 'moisture_mean', 'vegetation_mean', 'edge_mean']
    """
    
    if feature in ['entropy_sum', 'entropy_mean', 'entropy_max', 'entropy_min', 'entropy_std']:
      original_image = self._display_single_image(idx=idx, display=False, output=True, layer='rgb')[1]
      gray_scale_image = rgb2gray(original_image)
      entropy_of_image = entropy(gray_scale_image, disk(3))
      if feature in ['entropy_sum', 'entropy_max', 'entropy_min']:
        feature_nr = np.sum(entropy_of_image)
      elif feature == 'entropy_mean':
        feature_nr = np.mean(entropy_of_image)
      elif feature == 'entropy_std':
        feature_nr = np.std(entropy_of_image)
      if display:
        plt.title("Entropy feature map of 'tree_cover'")
        imshow(entropy_of_image, cmap='magma')
    
    elif feature in ['moisture_index_sum', 'moisture_index_mean', 'moisture_index_max', 'moisture_index_min', 'moisture_index_std']:
      original_image = self._display_single_image(idx=idx, display=False, output=True, layer='moisture_index')[1]
      entropy_of_image = original_image #entropy(original_image, disk(3))
      if feature in ['moisture_index_sum', 'moisture_index_max', 'moisture_index_min']:
        feature_nr = np.sum(entropy_of_image)
      elif feature == 'moisture_index_mean':
        feature_nr = np.mean(entropy_of_image)
      elif feature == 'moisture_index_std':
        feature_nr = np.std(entropy_of_image)
      if display:
        imshow(entropy_of_image, cmap='magma')

    elif feature in ['vegetation_index_sum', 'vegetation_index_mean', 'vegetation_index_max', 'vegetation_index_min']:
      original_image = self._display_single_image(idx=idx, display=False, output=True, layer='vegetation_index')[1]
      entropy_of_image = original_image #entropy(original_image, disk(3))
      if feature in ['vegetation_index_sum', 'vegetation_index_max', 'vegetation_index_min']:
        feature_nr = np.sum(entropy_of_image)
      elif feature == 'vegetation_index_mean':
        feature_nr = np.mean(entropy_of_image)
      elif feature == 'vegetation_index_std':
        feature_nr = np.std(entropy_of_image)
      if display:
        imshow(entropy_of_image, cmap='magma')
    
    elif feature in ['edge_sum', 'edge_mean', 'edge_max', 'edge_min', 'edge_std']:
      original_image = self._display_single_image(idx=idx, display=False, output=True, layer='rgb')[1]
      gray_image = rgb2gray(original_image)
      blurred_gray_image = cv2.GaussianBlur(gray_image, (3,3), 0)
      edges = edge_feature_library.canny(blurred_gray_image, sigma=1.5)
      if feature in ['edge_sum', 'edge_max', 'edge_min']:
        feature_nr = np.sum(edges)
      elif feature == 'edge_mean':
        feature_nr = np.mean(edges)
      elif feature == 'edge_std':
        feature_nr = np.std(edges)
      if display:
        plt.title("Gaussian blur applied on 'tree_cover'")
        plt.imshow(blurred_gray_image)#edges)

    elif feature in ['rgb_std', 'rgb_std_max', 'rgb_std_min']:
      layer_1 = float(torch.std(self.__getitem__(idx=idx, layers=[1])['image']))
      layer_2 = float(torch.std(self.__getitem__(idx=idx, layers=[2])['image']))
      layer_3 = float(torch.std(self.__getitem__(idx=idx, layers=[3])['image']))
      feature_nr = np.sum([layer_1, layer_2, layer_3])
      if display:
        warnings.warn('This feature has no display feature')

    else:
      raise ValueError('This feature in _get_single_feature() has not been implemented yet')
   
    if output:
      return (idx, feature_nr)

  def get_features(self, idx, display=False, output=True, feature='entropy_sum'):
    """
    get one feature of multiple images

    Parameters: 
      feature (str)          : The layer which will be displayed or returned. Only one layer may be selected at once
                'entropy_sum': the red, green, blue layers are combined to view the image as seen by humans
      display (bool)         : if True, display a summary of the distribution of features
      sensitiviy (float [0.5, 1]): the percentage of the picture which must be covered by a single category. 
                                   Where no category covers more than 50%, the image will be defined as 'mixed'
      steps (int)            : The number of groups the features will be distributed on. The higher the number, the higher the risk of a group being empty
      output (bool)          : if True, return a 
    """
    if isinstance(idx, int):
      return self._get_single_feature(idx, display=display, output=output, feature=feature)
    else:
      return list(map(lambda iterable: self._get_single_feature(iterable, display=display, output=output, feature=feature), idx))  
  

  def _get_features_of_category(self, idx=False, feature='entropy_sum', display=False, sensitivity=0, steps=20, output=True):
    """
    get the feature characteristics of multiple images per category

    Parameters: 
      idx (bool|list) : if False, return category of every img in self.frame. If list, return only for values inside list
      feature (str)          : The layer which will be displayed or returned. Only one layer may be selected at once
                'entropy_sum': the red, green, blue layers are combined to view the image as seen by humans
      display (bool)         : if True, display a summary of the distribution of features
      sensitiviy (float [0.5, 1]): the percentage of the picture which must be covered by a single category. 
                                   Where no category covers more than 50%, the image will be defined as 'mixed'
      steps (int)            : The number of groups the features will be distributed on. The higher the number, the higher the risk of a group being empty
      output (bool)          : if True, return a dic containing the categories distributed into steps {category:{step:[[idx1, value1], [idx2, value2]]}}
    """
    dic_containing_all_features = {}
    all_data_in_categories = self.assign_category(idx=idx, sensitivity=sensitivity, translated=True)
    list_containing_every_category = []
    list_containing_every_key = []
    for category in all_data_in_categories.keys():
      temp_category_list = all_data_in_categories[category]
      all_features_per_category = self.get_features(temp_category_list, feature=feature)
      all_features_per_category = sorted(all_features_per_category, key=lambda tupl: tupl[1])

      max_value = all_features_per_category[-1][1]
      min_value = all_features_per_category[0][1]
      step_length = (max_value - min_value)/steps
      temp_step = min_value
      dic_containing_all_steps = {}
      list_containing_one_step = []
      step_nr_lower = 0
      step_nr_upper = 100/steps
      for step in list(range(1, steps+1)):
        range_lower = temp_step
        range_upper = temp_step + step_length
        for tup in all_features_per_category:
          if range_lower <= tup[1] < range_upper:
            list_containing_one_step += [tup]
          if range_upper == max_value:
            list_containing_one_step += [all_features_per_category[-1]]

        label_temp = str(step_nr_upper) + '%' #str(step_nr_lower) + '%' + '-' + 
        dic_containing_all_steps[label_temp] = list(set(list_containing_one_step))
        temp_step += step_length
        list_containing_one_step = []
        step_nr_lower += 100/steps
        step_nr_upper += 100/steps
      dic_containing_all_features[category] = dic_containing_all_steps

      if display:
        new_list_containing_every_category = np.sum(list_containing_every_category, axis=0).tolist()
        list_containing_all_lengths = []
        list_containing_all_keys = []
        for key in dic_containing_all_features[category].keys():
          list_containing_all_lengths += [int(len(dic_containing_all_features[category][key]))]
          list_containing_all_keys += [str(key)]
        list_containing_every_category += [list_containing_all_lengths]
        list_containing_every_key = list_containing_all_keys
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(list_containing_all_keys, list_containing_all_lengths)
        plt.title('Number of values per percentile for ' + str(category))
        plt.xlabel('Percentile')
        plt.ylabel('Number of Values')
        plt.show()
    
    new_list_containing_every_category = np.sum(list_containing_every_category, axis=0).tolist()
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(list_containing_every_key, new_list_containing_every_category)
    plt.title('Number of values per percentile for the whole data set')
    plt.xlabel('Percentile')
    plt.ylabel('Number of Values')
    plt.show()


    if output:
      return dic_containing_all_features

  def get_mean_std_per_layer(self, idx, layers):
    """
    Get the mean and std for all images for every layer in sample or image
    """
    return None




  def train_val_test_split(self, train_size=0.6, seed=42, sensitivity=0, mode='native', translated=False, debug=False, minimum_input=50, reduce_train=0.1):
    """
    Return three lists containing the indices of the train_val_test data in the format train, val, test. As such call this method as follows: 
    train, val, test = self.train_val_split(parameters)

    Parameters:
      train_size (float (0, 1)): the percentage of all the data which will be used for the train set. The val/test set will split the remaining data in half
      seed (int): control the randomness of the split
      sensitiviy (float [0.5, 1]): the percentage of the picture which must be covered by a single category. 
                                   Where no category covers more than 50%, the image will be defined as 'mixed'
      mode (str): the mode which decides how the training data is split
                  'native': all the available data is split into train/val/test. This is the fastest complete method
                  'quick' : the first 128 pictures are assigned to train, the next 128 to val etc. 
                            This mode is purely for the debugging of functions and not to create accurate models
                  'r_min' : get a random selection of images based on the smallest category. I.e. if the smallest valid category contains 36 images, 
                            only 36 images will be taken from each category
                            Warning: The smaller the smallest valid category, the higher the risk of imbalanced val/test sets due to rounding errors
            'entropy_sum' : Get a random selection of values for each category based on reduce_train(affects length) and the entropy sum(affects returned images) of each image. 
                            This mode ensures the distribution of images based on the entropy_sum is the same as in the complete training set
            'entropy_mean': Get a random selection of values for each category based on reduce_train(affects length) and the entropy mean(affects returned images) of each image. 
                            This mode ensures the distribution of images based on the entropy_sum is the same as in the complete training set
                 'random' : Get a truly random selection of images, disregarding category distribution. Size depends on the reduce_train variable

      translated (bool): if True, get the category as string. Otherwise as int. False is recommended in this function
      debug (bool): get a more detailed view of all the steps
      minimum_input (int): the number of input images per category needed to be used by the train_val_test split. It is highly recommended to use a number >10
      reduce_train (float): the percentage you want to reduce the training data to. I.e. if 0.1, the train set will be reduced to 10% of its original size. Only applicable to mode ['entropy']
                           IMPORTANT: If a category has significantly less images than other categories, this may lead to slight inaccuracies
    """  
    all_data_in_categories = self.assign_category(sensitivity=sensitivity, translated=translated)
    temp_train, train, val, test = [], [], [], []
    for category in all_data_in_categories.keys():
      temp_category_list = all_data_in_categories[category]
      random.Random(seed).shuffle(temp_category_list)
      temp_list_len = len(temp_category_list)
      if temp_list_len>minimum_input: # If the dataset has only a handful of pictures for a category, the result will not be representative and risk breaking the code

        # This looks like an utter mess, I know. All this does, is saying that the first i.e. 60% of the list go into the test set, 
        # the next 20% into the val/test sets. Because the list is randomly shuffled beforehand, the distribution for each category is still random 
        # the floor method is used to ensure that range does not overlap. The consequence is miniscule variances between the val/test sets
        train_upper_limit = int(np.floor(temp_list_len*train_size))
        val_upper_limit   = train_upper_limit + int((((1-train_size)*temp_list_len)/2))
        temp_train       += temp_category_list[0:train_upper_limit]
        val              += temp_category_list[train_upper_limit:val_upper_limit]       
        test             += temp_category_list[val_upper_limit:temp_list_len]
        # The reason for assigning the val/test lists for each module here is to ensure that the train splits of different modes can be compared against each other

        if debug:
          print(category, temp_list_len)
          print('train', 0, int(np.floor(temp_list_len*train_size)))
          print('val', int(np.floor(temp_list_len*train_size)), int(np.floor(temp_list_len*train_size))+int((((1-train_size)*temp_list_len)/2)))
          print('test', int(np.floor(temp_list_len*train_size))+int((((1-train_size)*temp_list_len)/2)), temp_list_len)
          print('train', len(temp_train), 'val', len(val), 'test', len(test))
    
    if mode == 'native':
      train = temp_train
      
    elif mode == 'quick':
      train, val, test = temp_train[:128], val[:128], test[:128]
    
    elif mode == 'r_min':
      all_data_in_train_categories = self.assign_category(idx=temp_train, sensitivity=sensitivity, translated=translated)
      smallest_valid_category_len = len(temp_train) + 1
      # Get the length of the smallest valid category 
      for category in all_data_in_train_categories.keys():
        temp_category_list = all_data_in_train_categories[category]
        random.Random(seed).shuffle(temp_category_list)
        temp_list_len = len(temp_category_list)
        if temp_list_len>minimum_input: # If the dataset has only a handful of pictures for a category, the result will not be representative and risk breaking the code
          if temp_list_len < smallest_valid_category_len:
            smallest_valid_category_len = temp_list_len

      # Get images from each category based on the smallest valid category length
      for category in all_data_in_train_categories.keys():
        temp_category_list = all_data_in_train_categories[category]
        random.Random(seed).shuffle(temp_category_list) 
        upper_limit = smallest_valid_category_len   
        train += temp_category_list[0:upper_limit]
    
    elif mode == 'random':
      total_len = len(temp_train)
      required_images = int(total_len*reduce_train)
      train = random.Random(seed).sample(temp_train, k=required_images)

    elif mode in ['rgb_std', 'entropy_sum', 'moisture_index_sum', 'vegetation_index_sum', 'edge_sum', 'entropy_mean', 'moisture_mean', 'vegetation_mean', 'edge_mean']:
      
      # This ensures minimal rounding error loss for datasets of arbitrary size at the cost of some performance
      longest_category_len = 0
      category_len_dic = self.assign_category(idx=temp_train)
      for category in category_len_dic.keys():
        if longest_category_len < len(category_len_dic[category]): # The rounding error decreases with increasing sample size
          longest_category_len = len(category_len_dic[category])
      dic_containing_all_entropy = self._get_features_of_category(idx=temp_train, feature=mode, sensitivity=sensitivity, steps=longest_category_len)
      total_len = len(temp_train)
      temp_entropy_list = []
      for category in dic_containing_all_entropy.keys():
        total_items_per_category = 0

        # Get the length of category
        for step in dic_containing_all_entropy[category]:
          total_items_per_category += len(dic_containing_all_entropy[category][step])

        category_multiplicator = (total_items_per_category/total_len) * reduce_train 
        max_pics_per_category = category_multiplicator * total_len #the total no° of pictures each category gets
        steps_per_img = total_items_per_category/max_pics_per_category

        images_per_countable_group = []
        
        for step in dic_containing_all_entropy[category]:
          if len(images_per_countable_group) >= steps_per_img:
            temp_entropy_list += random.Random(seed).sample(images_per_countable_group, k=1)
            images_per_countable_group = []
            """
            if  0 < len(dic_containing_all_entropy[category][step]):
              for tup in dic_containing_all_entropy[category][step]:
                images_per_countable_group += [tup]
            """
          else:
            for tup in dic_containing_all_entropy[category][step]:
              images_per_countable_group += [tup]
              if len(images_per_countable_group) >= steps_per_img:
                temp_entropy_list += random.Random(seed).sample(images_per_countable_group, k=1)
                images_per_countable_group = []
          """
          if 0 <= (len(dic_containing_all_entropy[category][step]) * category_multiplicator) < 1.5:
            temp_entropy_list += dic_containing_all_entropy[category][step]
          elif 2 <= (len(dic_containing_all_entropy[category][step]) * category_multiplicator):
            temp_entropy_list += random.Random(seed).choices(dic_containing_all_entropy[category][step], k=int(len(dic_containing_all_entropy[category][step]) * category_multiplicator))
          """
      for i in temp_entropy_list:
        train += [i[0]]
    
    elif mode in ['rgb_std_max', 'rgb_std_min', 'entropy_max', 'entropy_min', 'moisture_index_max', 'moisture_index_min', 'vegetation_index_max', 'vegetation_index_min', 'edge_max', 'edge_min']:
      total_len = len(temp_train)
      temp_feature_list = []
      dic_containing_all_categories = self.assign_category(idx=temp_train, sensitivity=sensitivity)
      for category in dic_containing_all_categories.keys():
        number_of_pictures_for_category = int((reduce_train * (len(dic_containing_all_categories[category])/total_len)) * total_len)
        complete_feature_list = self.get_features(idx=dic_containing_all_categories[category], display=False, output=True, feature=mode)
        if mode in ['entropy_max', 'moisture_index_max', 'vegetation_index_max', 'edge_max', 'rgb_std_max']:
          complete_feature_list = sorted(complete_feature_list, key=lambda tup: tup[1], reverse=True)
          temp_feature_list += complete_feature_list[number_of_pictures_for_category::-1]
        elif mode in ['entropy_min', 'moisture_index_min', 'vegetation_index_min', 'edge_min', 'rgb_std_min']:
          complete_feature_list = sorted(complete_feature_list, key=lambda tup: tup[1])
          temp_feature_list += complete_feature_list[:number_of_pictures_for_category]
      for i in temp_feature_list:
        train += [i[0]]  
    
    else:
      raise ValueError('This mode has not been accounted for yet. Please try another supported input type')
    
    random.Random(seed).shuffle(train)
    return train, val, test

  def merge_sets(self, set_1, set_2, multiplier_1, seed=42):
    """
    Merge two lists based on a condition
    """
    final_list = []
    set_1_categories = self.assign_category(idx=set_1)
    for category in set_1_categories.keys():
      all_values_in_category = set_1_categories[category]
      pictures_per_category = int(multiplier_1 * len(all_values_in_category))
      final_list += random.Random(seed).sample(all_values_in_category, k=pictures_per_category)
    set_2_categories = self.assign_category(idx=set_2)
    for category in set_2_categories.keys():
      all_values_in_category = set_2_categories[category]
      pictures_per_category = int((1-multiplier_1) * len(all_values_in_category))
      final_list += random.Random(seed).sample(all_values_in_category, k=pictures_per_category)
      
    return final_list
