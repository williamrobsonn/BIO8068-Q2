#Question 2 BIO8068 Assessment

#Loading in the libraries needed ----

library(rinat)
library(sf)
library(keras)
library(ggplot2)

#Sourcing the files needed for the bulk data download and setting download limits (aka UK) ----

source("download_images.R") 
gb_ll <- readRDS("gb_simple.RDS")

#Now looking at the species of interest ----

#Looking for the common carder bumblebee, specifying the photos need to be of research quality 

carderbee_recs <-  get_inat_obs(taxon_name  = "Bombus pascuorum",
                                bounds = gb_ll,
                                quality = "research",
                                # month=6,   # Month can be set.
                                # year=2018, # Year can be set.
                                maxresults = 700)

#Now to bulk download these images (may take some time) 

download_images(spp_recs = carderbee_recs, spp_folder = "carder")

#Now to do the same for the other species ----

#Red tailed bumblebees 

redtailed_recs <-  get_inat_obs(taxon_name  = "Bombus lapidarius",
                                bounds = gb_ll,
                                quality = "research",
                                # month=6,   # Month can be set.
                                # year=2018, # Year can be set.
                                maxresults = 700)

#Downloading the images 

download_images(spp_recs = redtailed_recs, spp_folder = "redtailed")

#Buff tailed bumblebees

bufftailed_recs <-  get_inat_obs(taxon_name  = "Bombus terrestris",
                                bounds = gb_ll,
                                quality = "research",
                                # month=6,   # Month can be set.
                                # year=2018, # Year can be set.
                                maxresults = 700)

#Downloading the images

download_images(spp_recs = bufftailed_recs, spp_folder = "bufftailed")

#We now need to put some of the images into a test folder ----

image_files_path <- "images" # path to folder with photos

# list of spp to model; these names must match folder names
spp_list <- dir(image_files_path) # Automatically pick up names
#spp_list <- c("brimstone", "hollyblue", "orangetip") # manual entry

# number of spp classes (i.e. 3 species in this example)
output_n <- length(spp_list)

# Create test, and species sub-folders
for(folder in 1:output_n){
  dir.create(paste("test", spp_list[folder], sep="/"), recursive=TRUE)
}

# Now copy over spp_601.jpg to spp_700.jpg using two loops, deleting the photos
# from the original images folder after the copy
for(folder in 1:output_n){
  for(image in 601:700){
    src_image  <- paste0("images/", spp_list[folder], "/spp_", image, ".jpg")
    dest_image <- paste0("test/"  , spp_list[folder], "/spp_", image, ".jpg")
    file.copy(src_image, dest_image)
    file.remove(src_image)
  }
}

#Training the deep learning model ----

# image size to scale down to (original images vary but about 400 x 500 px)
img_width <- 150
img_height <- 150
target_size <- c(img_width, img_height)

# Full-colour Red Green Blue = 3 channels
channels <- 3

#Rescaling images and setting max hue

# Rescale from 255 to between zero and 1
train_data_gen = image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

#Reading the images from the folder ----

#We call this step twice, one for training and one for validation 

# training images
train_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "training",
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "validation",
                                                    seed = 42)


#Checking to see if the right number of classes have been read in 

cat("Number of images per class:")


table(factor(train_image_array_gen$classes))

cat("Class labels vs index mapping")

train_image_array_gen$class_indices

#To view an image 
plot(as.raster(train_image_array_gen[[1]][[1]][8,,,])) #Buff tailed was displayed for me 

#Defining additional parameters and configuring the model ----

# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32 # Useful to define explicitly as we'll use it later
epochs <- 10     # How long to keep training going for (might increase)

#Now lets make the model

# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), input_shape = c(img_width, img_height, channels), activation = "relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), activation = "relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n, activation = "softmax") 


print(model) #looks all okay here

# Compile the model
#Categorical crossentropy is used as we have more than one species

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

#Save before running this last step ----

# Train the model with fit_generator
history <- model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2
)

#Assessing the accuracy and loss ----

plot(history) 
#Appears to be a slight over-training after 6 epochs. Proven by training loss continuing to decline, whilst validation
#has a lower loss rate. Validation is also low at 55%!

#Saving the model for later use and image for other use ----

# The imager package also has a save.image function, so unload it to
# avoid any confusion
detach("package:imager", unload = TRUE)

# The save.image function saves your whole R workspace
save.image("bees.RData")

# Saves only the model, with all its weights and configuration, in a special
# hdf5 file on its own. You can use load_model_hdf5 to get it back.
#model %>% save_model_hdf5("animals_simple.hdf5")


#Testing the model ----

path_test <- "test"

test_data_gen <- image_data_generator(rescale = 1/255)

test_image_array_gen <- flow_images_from_directory(path_test,
                                                   test_data_gen,
                                                   target_size = target_size,
                                                   class_mode = "categorical",
                                                   classes = spp_list,
                                                   shuffle = FALSE, # do not shuffle the images around
                                                   batch_size = 1,  # Only 1 image at a time
                                                   seed = 123)

# Takes about 3 minutes to run through all the images
model %>% evaluate_generator(test_image_array_gen, 
                             steps = test_image_array_gen$n)

#Accuracy at 53% so quite low again, lower than the initial model! 

#Viewing observed and expected for our images  ----

predictions <- model %>% 
  predict_generator(
    generator = test_image_array_gen,
    steps = test_image_array_gen$n
  ) %>% as.data.frame
colnames(predictions) <- spp_list

# Create 3 x 3 table to store data
confusion <- data.frame(matrix(0, nrow=3, ncol=3), row.names=spp_list)
colnames(confusion) <- spp_list

obs_values <- factor(c(rep(spp_list[1],100),
                       rep(spp_list[2], 100),
                       rep(spp_list[3], 100)))
pred_values <- factor(colnames(predictions)[apply(predictions, 1, which.max)])

library(caret)
conf_mat <- confusionMatrix(data = pred_values, reference = obs_values)
conf_mat

#Carder bee was the best at being identified, but was relatively low for the negatives identified. The other two species are the opposite!



