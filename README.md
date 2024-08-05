# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

# Mildew Dectection in Cherry Leaves

## Project Description

Marianne McGuineys, head of IT and Innovation at Farmy & Foods, is facing a challenge with their cherry plantations presenting powdery mildew, a fungal disease affecting a wide range of plants. Cherry plantation is one of the finest products in the company's portfolio, and they are concerned about supplying the market with a compromised quality product.

Currently, the process involves manually verifying if a cherry tree contains powdery mildew, where an employee spends around 30 minutes per tree taking leaf samples and visually inspecting them. If mildew is detected, a specific compound is applied to kill the fungus, which takes about 1 minute. Given the thousands of cherry trees across multiple farms, this manual process is not scalable due to the time required for inspection.

To save time, the IT team suggested developing a Machine Learning (ML) system capable of instantly detecting whether a cherry leaf is healthy or has powdery mildew using an image of the leaf. A similar process is in place for other crops for pest detection, and success in this initiative could lead to replicating the project for all other crops. The dataset comprises cherry leaf images provided by Farmy & Foods, taken from their crops.

This project aims to build a dashboard to detect whether a cherry leaf is healthy or has powdery mildew.


## Dataset Content

The dataset contains 4,208 images of cherry leaves, which are either healthy or infested with powdery mildew. These images were taken from the crops of Farmy & Foods. Each image has dimensions of 256x256 pixels. The dataset is labeled into two classes: healthy and powdery mildew.

You can access the dataset on [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).


## Business Requirements

Farmy & Foods, a company in the agricultural sector, has requested the development of a Machine Learning-based system to instantly detect powdery mildew in cherry trees. This system should use images of cherry leaves to determine whether they are healthy or require treatment. The current manual inspection process, which takes about 30 minutes per tree, is not scalable due to the large number of cherry trees spread across multiple farms. Automating this process with an ML system aims to save time and improve efficiency.

Summary:
1. Conduct a study to visually differentiate a healthy cherry leaf from one infected by powdery mildew.
2. Predict if a cherry tree is healthy or contains powdery mildew.
3. Obtain a prediction report of the examined leaves.

## Hypothesis and Validation

### Hypothesis 1
**Hypothesis:** Infected leaves have clear visual markers differentiating them from healthy leaves.
**Validation:** Conduct a visual study of average images and variability images for each class (healthy and powdery mildew). Create image montages to highlight these differences.

### Hypothesis 2
**Hypothesis:** A neural network with specific architectural choices (e.g., using softmax  in the output layer) will yield better classification performance.
**Validation:** Train models using softmax configurations and optimize performance metrics, such as accuracy and loss, using learning curves and validation results.

### Hypothesis 1

**Hypothesis:** Infected leaves have distinct visual features that differentiate them from healthy leaves.

#### Introduction

Cherry leaves affected by powdery mildew typically exhibit specific symptoms: initially, light-green, circular lesions appear on either leaf surface, followed by a subtle white, cotton-like growth in the infected areas. These characteristics must be translated into machine learning terms, requiring images to be properly prepared for optimal feature extraction and model training.

#### Understanding the Problem and Mathematical Functions

When working with an image dataset, normalizing the images before training a neural network is crucial for two main reasons:

1. **Consistency:** It ensures that the neural network produces consistent results when tested with new images.
2. **Transfer Learning:** Normalization facilitates transfer learning by standardizing the input images.

To normalize an image, we need to compute the mean and standard deviation of the entire dataset. This process considers four dimensions of an image:

- **B:** Batch size (number of images)
- **C:** Number of channels in the image (3 for RGB images)
- **H:** Height of each image
- **W:** Width of each image

The mean and standard deviation are calculated separately for each channel. Since loading the entire dataset into memory at once is impractical, we can load small batches of images sequentially to compute these statistics, which can be a non-trivial task.

2. Observation
A visual montage highlights the clear distinctions between healthy leaves and those infected with mildew.


There is no visible difference betweeen avaerage and variabilty images to enable the identification of affected leaves.

3. Conclusion
The model successfully identified the differences, learning to distinguish and generalize for accurate predictions. A well-trained model predicts classes on a data batch without overfitting, enabling it to generalize and accurately forecast future observations by recognizing general patterns rather than memorizing specific relationships from the training dataset.

### Hypothesis 2

**Hypothesis:** Using Sigmoid activation function in the CNN output layer for this classification task.

#### Introduction

Understanding the Problem and Mathematical Functions
The model needs to classify cherry leaves as either healthy or infected, making this a classification problem. This can be approached as binary classification (healthy vs. infected).

For binary classification, a single output node is used, with probabilities ranging from 0 to 1. If the probability is less than 0.5, the leaf is classified as healthy (class 0); if it is 0.5 or greater, it is classified as infected (class 1). This is managed by the sigmoid function, which compresses outputs to the [0, 1] range. However, sigmoid functions can suffer from sharp damp gradients during backpropagation, affecting learning efficiency.

Backpropagation adjusts network weights by flowing error values back through the network, using the derivative of the sigmoid function to inform weight adjustments. However, very high or low error values result in low derivatives, causing a "squashing" effect.

For multiclass classification with two output nodes, the softmax function is used. This function also outputs values in the [0, 1] range but normalizes the outputs into a probability distribution that sums to 1. The class with the highest probability is chosen.

2. Obseervation
The model training output is depicted in two graphs: one for loss and one for accuracy, both showing training and validation metrics.

#### Loss Graph:
1. **Initial Decrease**: Both training and validation loss start high and decrease significantly in the first few epochs, indicating effective learning.
2. **Fluctuations**: Validation loss shows fluctuations, especially around epochs 3 and 13, suggesting potential overfitting.
3. **Convergence**: Training and validation losses converge towards lower values overall, indicating positive training progress.

#### Accuracy Graph:
1. **Initial Increase**: Both training and validation accuracies increase rapidly in the initial epochs, demonstrating quick learning.
2. **High Values**: Training and validation accuracies reach and stay above 0.95 for most of the training process, indicating good performance.
3. **Fluctuations**: Validation accuracy shows some fluctuations, similar to validation loss, indicating possible overfitting or variability in the validation set.

### Conclusion:
The model exhibits strong learning capabilities and high accuracy, but shows signs of overfitting. Adjusting regularization and validation strategies should help stabilize the validation metrics and improve overall performance.



#### Understanding How to Evaluate Performance

Learning curves plot model performance over time or experience, typically showing the model's progress on both training and validation datasets. They help diagnose issues like underfitting or overfitting. 

Key terms include:
- **Epoch:** One complete pass through the training data.
- **Loss:** Measures prediction errors. Lower loss indicates better performance.
- **Accuracy:** Fraction of correct predictions. Higher accuracy indicates better performance.

A good fit is shown by decreasing training and validation loss to a stable point with minimal gap. A small generalization gap between training and validation loss/accuracy indicates good performance. Overfitting occurs if training continues beyond this point, which is why early stopping is often used.


### Rationale for the Model

The model is designed to efficiently classify cherry leaves as healthy or infected with powdery mildew. Below are the architectural choices and their rationale:

The model has one input layer, three hidden layers (three convolutional layers, one fully connected layer), and one output layer.

#### Goal

Hyperparameters, the number of hidden layers, and the optimizer were chosen through trial and error. While this model might not be the ultimate best, it was selected based on the performance observed through multiple tests and adjustments. This model balances the ability to generalize and predict accurately without overfitting, as it learns the general patterns from the dataset rather than memorizing specific relationships.

#### Choosing the Hyperparameters

- **Convolutional Layer Size:** A 2D CNN (Conv2D) is suitable for our non-volumetric images. A 1D convolution layer is not appropriate here.
- **Convolutional Kernel Size:** The 3x3 convolutional filter moves across the image dimensions, capturing small details effectively. It is preferred over 2x2 and 5x5 for its balance in capturing fine details and computational efficiency.
- **Number of Neurons:** Powers of 2 are chosen for computational efficiency and optimization.
- **Activation Function:** ReLU is chosen for its simplicity, speed, and ability to avoid the vanishing gradient problem, helping the network converge quickly and reliably.
- **Pooling:** MaxPooling is used to reduce computational complexity and variance, selecting the most prominent features. It works well with our dataset where the background is dark and mildew is lighter.
- **Output Activation Function:** Sigmoid is used for binary classification.
- **Dropout:** A dropout rate of 50% prevents overfitting by nullifying some neurons during training, ensuring that the model generalizes well.

## Rationale for Mapping Business Requirements to Data Visualizations and ML Tasks

The primary business requirements were translated into user stories, which were then broken down into machine learning tasks. Each task was manually tested to ensure proper functionality.

### Business Requirement 1: Data Visualization

The client needs a study to visually distinguish between a healthy cherry leaf and one affected by powdery mildew.

#### User Stories:
- As a client, I want to navigate easily through an interactive dashboard to view and understand the presented data.
- As a client, I want to see the "mean" and "standard deviation" images for both healthy and powdery mildew-infected cherry leaves to visually differentiate them.
- As a client, I want to see the differences between an average healthy cherry leaf and an average infected leaf.
- As a client, I want to see an image montage for both healthy and infected cherry leaves for visual comparison.

These user stories are addressed by implementing the following tasks in the Streamlit dashboard and the data visualization notebook:
- A Streamlit-based dashboard with an easy-to-navigate sidebar.
- Visualization of differences between average healthy and infected leaves.
- Display of "mean" and "standard deviation" images for both classes.
- Creation of image montages for healthy and infected leaves.

### Business Requirement 2: Classification

The client wants to determine if a given cherry leaf is affected by powdery mildew.

#### User Story:
- As a client, I want a machine learning model to predict with at least 86% accuracy whether a cherry leaf is healthy or infected.

This user story is addressed by the following tasks presented in the Streamlit dashboard and data visualization notebook:
- Explanation of the rationale behind the deployed ML model.
- An uploader widget allowing the client to upload cherry leaf images for instant evaluation. Key features include:
  - Images must be in .jpeg format.
  - Multiple images can be uploaded at once, up to 200MB.
  - The dashboard displays the uploaded image and the prediction, indicating if the leaf is infected with powdery mildew and the associated probability.

### Business Requirement 3: Report

The client desires a prediction report of the examined leaves.

#### User Story:
- As a client, I want to obtain a report from the ML predictions on new leaves.

This user story is addressed by implementing the following task in the Streamlit dashboard:
- After each batch of images is uploaded, a downloadable .csv report with the predicted statuses is made available.


## ML Business Case

- In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.

## Dashboard Design

- List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items, that your dashboard library supports.
- Finally, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project, you were confident you would use a given plot to display an insight, but later, you chose another plot type).

## Unfixed Bugs

- You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.

## Acknowledgements (optional)

- Thank the people who provided support throughout this project.
