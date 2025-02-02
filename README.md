<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Food Classifier - Deep Learning Project</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f8f8f8;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        h1, h2 {
            text-align: center;
            color: #333;
        }
        img {
            display: block;
            margin: 0 auto;
            width: 80%;
        }
        .badge-container {
            text-align: center;
            margin: 10px 0;
        }
        .badge {
            background: #333;
            color: white;
            padding: 8px 15px;
            border-radius: 5px;
            margin: 5px;
            display: inline-block;
        }
        code {
            background: #eee;
            padding: 5px;
            border-radius: 5px;
            display: block;
            margin: 10px 0;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            font-size: 14px;
            color: #777;
        }
    </style>
</head>
<body>

    <div class="container">
        <h1>🍔 Food Classifier with TensorFlow & Keras</h1>
        <p align="center">A Deep Learning model for food image classification using <strong>TensorFlow</strong> and <strong>Keras</strong>. The model is trained on the <strong>Food-101 dataset</strong> and can be expanded with custom datasets.</p>

        <!-- Project Banner -->
        <img src="https://your-image-url.com/banner.png" alt="Food Classifier Banner">

        <!-- Badges -->
        <div class="badge-container">
            <span class="badge">TensorFlow 2.10+</span>
            <span class="badge">Python 3.8+</span>
            <span class="badge">MIT License</span>
        </div>

        <h2>🚀 Features</h2>
        <ul>
            <li>✅ Deep Learning Model using CNN</li>
            <li>✅ Food-101 Dataset Integration</li>
            <li>✅ Data Augmentation for Better Accuracy</li>
            <li>✅ Saves Model (<code>.h5</code> file) for Future Use</li>
            <li>✅ Custom Datasets Supported</li>
        </ul>

        <h2>📂 Dataset Structure</h2>
        <pre>
/food_dataset/
  /pizza/
    - img1.jpg
    - img2.jpg
  /burger/
    - img1.jpg
    - img2.jpg
  /apple/
    - img1.jpg
    - img2.jpg
        </pre>
        <p>If using <strong>Food-101</strong>, ensure an 80-20 train-validation split.</p>

        <h2>🛠️ Installation</h2>
        <p><strong>Step 1:</strong> Clone the repository</p>
        <code>git clone https://github.com/your-username/Food-Classifier.git</code>

        <p><strong>Step 2:</strong> Install dependencies</p>
        <code>pip install tensorflow numpy matplotlib opencv-python</code>

        <p><strong>Step 3:</strong> Run the training script</p>
        <code>python train.py</code>

        <h2>📊 Model Performance</h2>
        <table border="1" cellpadding="5" cellspacing="0" width="100%">
            <tr>
                <th>Epoch</th>
                <th>Training Accuracy</th>
                <th>Validation Accuracy</th>
            </tr>
            <tr>
                <td>1</td>
                <td>14.5%</td>
                <td>27.4%</td>
            </tr>
            <tr>
                <td>2</td>
                <td>23.8%</td>
                <td>28.0%</td>
            </tr>
            <tr>
                <td>3</td>
                <td>26.1%</td>
                <td>30.2%</td>
            </tr>
        </table>

        <h2>📸 Sample Predictions</h2>
        <img src="https://your-image-url.com/sample_predictions.png" alt="Food Classifier Predictions">

        <h2>🎯 Future Improvements</h2>
        <ul>
            <li>✅ Add More Training Data (e.g., Food-101)</li>
            <li>✅ Improve Model with Transfer Learning</li>
            <li>✅ Deploy Model as a Web App</li>
        </ul>

        <h2>📝 License</h2>
        <p>This project is licensed under the <strong>MIT License</strong>.</p>

        <h2>⭐ Like This Project?</h2>
        <p>If you find this useful, give it a ⭐ on <a href="https://github.com/your-username/Food-Classifier" target="_blank">GitHub</a>!</p>

        <div class="footer">
            <p>© 2024 Food Classifier | Built with ❤️ by <a href="https://github.com/your-username" target="_blank">Your Name</a></p>
        </div>
    </div>

</body>
</html>
