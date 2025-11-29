Object recognition is a key challenge in the field of computer vision, and it has a wide range of 
applications, including robotics, self-driving cars, surveillance, and augmented reality. Over the years, two 
main approaches have emerged to address this challenge. The first is traditional computer vision (CV), 
which relies on handcrafted features and classical machine learning algorithms. The second is deep 
learning, which utilizes convolutional neural networks (CNNs) to learn features directly from data.  
In this coursework, we’ll be comparing the performance of these two approaches in object recognition 
tasks. We’ll be using two distinct datasets for our analysis: CIFAR-10 and the RGB-D Object Dataset. 
CIFAR-10 is a well-known benchmark that includes 60,000 labelled RGB images across 10 different 
object categories, making it a great resource for image classification. On the other hand, the RGB-D Object 
Dataset presents a more intricate challenge, containing over 300 object instances across 51 categories, with 
RGB-D frames captured in real-world environments. 
The goal of our comparison is to explore the trade-offs between traditional computer vision methods and 
deep learning techniques. We’ll look at various factors, including accuracy, computational efficiency, 
interpretability, and data requirements. Traditional methods, such as the Bag of Visual Words (BoVW) 
combined with Support Vector Machines (SVMs), tend to be lightweight and interpretable. However, they 
often struggle to handle complex visual variations. Deep learning models, particularly CNNs, usually 
deliver better performance but come with the need for large datasets, extensive computational resources, 
and precise tuning. 
