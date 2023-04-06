def preprocessing_data(path, main_folder, testing_folder):
    ds_train = image_dataset_from_directory(path + main_folder + testing_folder,
    color_mode="rgb", batch_size=64, label_mode="binary",shuffle=True, seed=0, 
    interpolation="bilinear" )
    resize_and_rescale = Sequential([
        layers.Resizing(254, 254 , interpolation = 'bilinear'),
        layers.Rescaling(1./255)], ### importante
        name='resize_and_rescale')
    augmentation = Sequential([layers.RandomFlip(), 
                           layers.RandomRotation(factor=0.05), 
                           #Rotation adjuted to +/- 0.05 as the pictures are mainly faces, and its very improbable to have a
                           #full 180 degrees rotation, but it's important to have a slight rotation considering photograph prespective
                           layers.RandomZoom(height_factor=0.1, width_factor=0.1),
                           layers.RandomContrast(factor=0.25),
                           layers.RandomBrightness(factor=0.2),
                           layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1))], 
                           name="my_augmentation_pipeline")
    ds_train = ds_train.map(lambda x, y: (resize_and_rescale(x), y))
    ds_train_augmented = ds_train.map(lambda x, y: (augmentation(x), y))
    train_size = int(0.8 * ds_train_augmented.cardinality().numpy())
    val_size = ds_train_augmented.cardinality().numpy() - train_size
    ds_train = ds_train_augmented.take(train_size)
    ds_val = ds_train_augmented.skip(train_size).take(val_size)
    return ds_train, ds_val

    
    