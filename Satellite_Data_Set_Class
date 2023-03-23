class Satellite_Data_Set_Class(torch.utils.data.Dataset):


  def __init__(self, root_dir, csv_file_dir=False, csv_file=False, transform=None):
    """
    Parameters:
      csv_file_dir (str): path to the csv file
      csv_file (csv): alternative to csv_file_dir; used if modifications to csv_file are necessary
      root_dir (str): path to the directory containing all files
      transform (callable): optional transformer
    """    

    if isinstance(csv_file_dir, str):
      self.frame = pd.read_csv(csv_file).copy() #pointer
    elif isinstance(csv_file, pd.DataFrame):
      self.frame = csv_file
    else:
      raise ValueError('The csv file has not been added correctly or is missing')
    self.root_dir = root_dir
    self.transform = transform

  # return a list containing all keys contained in the csv file
  def get_keys(self):
    list_with_keys = list(self.frame.columns[2:])
    list_with_keys.append('mixed')
    return list_with_keys

  # alternative length function as the torch implementation may struggle
  def len(self):
    return len(self.frame)

  # iterates over the input data and returns a list containing dic which contain the image/labels
  def get_item(self, idx, transform=True):
    """
    Parameters:
      idx (int, list, tensor): position of required image in the csv file
      transform (callable): optional transformer
    """    
    if torch.is_tensor(idx):
      idx = idx.tolist()

    if isinstance(idx, int):
      sample = self._get_single_item(idx, transform=transform)
    else:
      # This function returns a list containing dic with the data of all samples
      # the map function is used due to its performance increase over for loops
      # the use of the lambda function is necessary due to the optional parameters in the _get_single_item() function
      sample = list(map(lambda iterable: self._get_single_item(iterable, transform=transform), idx))    
    
    return sample

  # return the loaded .npy file with labels in a dictionary
  def _get_single_item(self, idx, transform=True):
    """
    This is an internal function, capable of getting only a single item at a time. This function is used in sds.get_item()

    Parameters:
      idx (int): position of required image in the csv file
      transform (callable): optional transformer
    """    

    # get the image, load it, then get the labels. Save all the data in a dic
    img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0]) 
    image = np.load(img_name).copy()
    labels = self.frame.iloc[idx, 2:]
    labels = np.array([labels])
    labels = labels.astype('float').reshape(-1, 11)
    sample = {'image': image, 'label': labels}

    if transform:
      if self.transform:
        sample['image'] = self.transform(sample['image'])
      else:
        warnings.warn('There was no transformer added')

    return sample

  def get_category(self, idx, translated=False, sensitivity=0.5):
    """
    Parameters:
      idx (int, list): position of required images category in the csv file
      translated (bool): if True, get the category as string. Otherwise as int
      sensitiviy (float [0.5, 1]): the percentage of the picture which must be covered by a single category
    """        
    if isinstance(idx, int):
      sample_list = self._get_category(idx, translated=translated, sensitivity=sensitivity)
    else:
      sample_list = list(map(lambda iterable: self._get_category(iterable, translated=translated, sensitivity=sensitivity), idx))
    return sample_list

  def _get_category(self, idx, translated=False, sensitivity=0.5):
    """
    Internal Function, able to get the category of a single image at a time based off of its position in the csv file

    Parameters:
      idx (int): position of required images category in the csv file
      translated (bool): if True, get the category as string. Otherwise as int
      sensitiviy (float [0.5, 1]): the percentage of the picture which must be covered by a single category
    """    

    list_containing_all_categories = [self.frame.iloc[idx, 2:]]
    if np.max(np.array(list_containing_all_categories))<sensitivity or np.max(np.array(list_containing_all_categories)) == sensitivity:
      if translated:
        result = 'mixed'
      else:
        result = 11
    else:
      index_of_value_making_up_majority_of_image = np.where(np.array(list_containing_all_categories)>sensitivity)[1][0]
      if translated:
        result = self.get_keys()[index_of_value_making_up_majority_of_image]
      else:
        result = index_of_value_making_up_majority_of_image
    return result

    # transform the data type of an image (requried for certain special functions)
  def change_type(self, idx, in_type, out_type, transform=True):
    """
    Parameters:
      idx (int, list, tensor): position of required image in the csv file
      in_type (str): the input type of the data based off the datas ending, i.e. "npy"
      out_type (str): the output type of the data based off the data ending i.e. "png"
      transform (callable): optional transformer
    """  
    image = self.get_item(idx, transform)['image']
    if in_type == 'npy':
      print('test')
    else:
      raise ValueError('This input_type has not been accounted for yet. Please try another supported input type')
    if out_type == 'npy':
      print('test')
    else:
      raise ValueError('This output_type has not been accounted for yet. Please try another supported output type')
  def display_image(self, idx):
    image, label = sds.get_item(self, idx).values()
