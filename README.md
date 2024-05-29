# Brain Tumor Detection and Classification Using Deep Neural Network and XAI Techniques

## Abstract
This project addresses the need for accurate brain tumor detection using MRI imaging. Manual interpretation is labor-intensive and error-prone, so we propose an automated approach using deep neural networks (DNNs), specifically convolutional neural networks (CNNs), for improved diagnostic efficiency and accuracy. We integrate explainable artificial intelligence (XAI) methodologies to enhance the interpretability of our model. This combination aims to provide medical professionals with a reliable tool for brain tumor identification and classification, improving patient care and treatment outcomes.

## Introduction
Brain tumors are a significant global health concern, encompassing a variety of subtypes that differ in their malignancy and impact on patient health. Early and accurate diagnosis is crucial for effective treatment planning and improving patient outcomes. However, the traditional approach to diagnosing brain tumors relies heavily on manual interpretation of MRI scans by radiologists, a process that is both time-consuming and prone to human error. This can lead to delays in diagnosis and treatment, potentially affecting patient prognosis.

To address these challenges, automated techniques leveraging deep neural networks (DNNs) have emerged as a promising solution. Convolutional neural networks (CNNs), a subset of DNNs, are particularly well-suited for image recognition tasks due to their ability to capture intricate patterns and features in medical images. Despite their high accuracy, one of the main drawbacks of these models is their lack of interpretability, which is critical in clinical settings where understanding the decision-making process is essential for trust and reliability.

This project aims to develop a robust and interpretable framework for brain tumor detection and classification using CNNs. By integrating explainable artificial intelligence (XAI) techniques, we seek to make the model's decision-making process transparent, thereby increasing its utility in clinical practice and aiding healthcare professionals in making informed decisions.

## Proposed Work
In this research project, we propose a comprehensive framework for automated brain tumor diagnosis that combines the power of deep neural networks (DNNs) with the transparency provided by explainable artificial intelligence (XAI) methodologies. Our approach involves several key steps:

1. **Data Collection and Preprocessing:**
   - We utilize a diverse dataset of MRI images from Kaggle, ensuring it includes various types and stages of brain tumors. The dataset consists of 7,023 MRI images categorized into four classes: Glioma, Meningioma, Pituitary, and No Tumor.
   - Preprocessing steps include image resizing to ensure uniform input dimensions, normalization to scale pixel values consistently, and data augmentation to increase the diversity of the training dataset.

2. **Model Development:**
   - We design and train a convolutional neural network (CNN) architecture tailored for medical imaging. The model incorporates Squeeze-and-Excitation (SE) blocks, which enhance feature discrimination by adaptively recalibrating feature maps.
   - The CNN is trained on the preprocessed MRI images, with hyperparameters fine-tuned using a validation set to prevent overfitting and ensure robustness.

3. **Integration of XAI Techniques:**
   - To address the issue of model interpretability, we integrate Gradient-weighted Class Activation Mapping (Grad-CAM) into our framework. Grad-CAM generates heatmaps that highlight regions of the MRI images that are most influential in the model's decision-making process.
   - These heatmaps provide visual explanations of the model's predictions, enabling medical professionals to understand and trust the model's outputs.

4. **Evaluation and Validation:**
   - The performance of the proposed model is evaluated using a testing set, with metrics such as accuracy, precision, recall, and F1-score calculated to assess its effectiveness.
   - The model achieves an overall accuracy of 96.19%, demonstrating its capability to accurately classify brain tumor images. The integration of Grad-CAM enhances interpretability, making the model's predictions more transparent and trustworthy.

5. **Clinical Applicability:**
   - By providing a reliable and interpretable tool for brain tumor diagnosis, this research has the potential to improve clinical decision-making and patient outcomes. The proposed framework can assist radiologists in identifying and classifying brain tumors more efficiently, reducing the likelihood of diagnostic errors and enabling timely treatment interventions.

## Data Description
The dataset consists of MRI scan images collected from Kaggle, divided into training and testing sets with labels for four classes: Glioma, Meningioma, Pituitary, and No Tumor.

### Image Distribution
| Class      | Training Set | Testing Set |
|------------|---------------|-------------|
| Glioma     | 1,321         | 300         |
| Meningioma | 1,339         | 306         |
| Pituitary  | 1,457         | 300         |
| No Tumor   | 1,595         | 405         |

## Data Preprocessing
- **Image Resizing:** Ensures consistent input dimensions.
- **Normalization:** Scales pixel values to a uniform range.
- **Data Augmentation:** Enhances dataset diversity by applying transformations.

## Experimental Analysis
The proposed model achieved an accuracy of 96.19%, with high precision, recall, and F1-score across all classes. Grad-CAM heatmaps provide interpretability, highlighting significant regions in MRI images.

### Accuracy Metrics
| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Glioma     | 0.89      | 0.99   | 0.94     | 150     |
| Meningioma | 0.98      | 0.86   | 0.91     | 153     |
| No Tumor   | 0.99      | 1.00   | 0.99     | 203     |
| Pituitary  | 0.99      | 0.99   | 0.99     | 150     |
| **Accuracy**|           |        | 0.96     | 656     |

## Conclusion
This project demonstrates the effectiveness of CNNs with SE blocks and XAI techniques for brain tumor classification, achieving a high accuracy of 96.19%. The integration of Grad-CAM enhances interpretability, providing valuable insights for clinical decision-making.

## Acknowledgments
I acknowledge Kaggle for the MRI dataset used in this research.

## License
This project is licensed under the MIT License.

