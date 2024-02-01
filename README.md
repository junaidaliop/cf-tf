# Recommendation Systems with TensorFlow


This repository offers a comprehensive guide to building recommendation systems using TensorFlow, particularly leveraging the MovieLens 100K and 1M datasets. It is crafted as an educational toolkit, featuring notebooks that are accessible and efficient for CPU use, without requiring GPU acceleration. These notebooks cover various recommendation system techniques, from matrix factorization to autoencoders, providing a practical, hands-on learning experience. This project aims to demystify the complexities of recommendation systems, making it an ideal starting point for those new to the field or looking to enhance their understanding of collaborative filtering and neural network applications in recommendation systems.
## Getting Started

Embark on your journey with our TensorFlow-based recommendation models by setting up your environment to run our comprehensive notebooks. These instructions will guide you through preparing your system to use our datasets and models, which are designed for efficient CPU-only execution. Whether you're delving into machine learning for the first time or looking to explore recommendation systems further, follow these steps to begin.

### Prerequisites

Before starting, ensure your system meets the following requirements:
- **Anaconda or Miniconda:** Essential for managing environments and dependencies, allowing for a smooth setup process.
- **Python 3.8 or higher:** Our notebooks are compatible with Python 3.8+, ensuring you can take advantage of the latest features and improvements.

This project is built on **TensorFlow 2.15.0** and **Keras 2.15.0**, among other dependencies. These can be found in the provided `cf-tf.yml` for environment setup or `requirements.txt` for direct installation.

### Environment Setup

1. **Clone the Repository:**
   Start by cloning the repository to your local machine. 
   ```bash
   git clone https://github.com/junaidaliop/cf-tf.git
   ```
2. **Navigate to the Project Directory:**
   Change into the project directory to begin setting up your environment.
   ```bash
   cd cf-tf
   ```

3. **Create the Conda Environment:**
   Use the `cf-tf.yml` file located in the `environment` directory to create a Conda environment. This file contains all necessary dependencies, ensuring a matched setup.
   ```bash
   conda env create -f environment/cf-tf.yml
   ```
   After creation, activate the environment:
   ```bash
   conda activate cf-tf
   ```

   Alternatively, if you prefer using `pip` and the `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Getting the Data

The project utilizes MovieLens datasets located in the `data` directory:
- **MovieLens 100K**: Stored in `data/MovieLens_100K/`
- **MovieLens 1M**: Stored in `data/MovieLens_1M/`

These datasets are essential for running the notebooks and testing the models.

### Running the Notebooks

Navigate to the `notebooks` directory to explore various recommendation models:
```bash
cd notebooks
```
Launch Jupyter Notebook or Jupyter Lab to open and run the notebooks:
```bash
jupyter notebook
# or
jupyter lab
```

Choose a notebook to begin, such as `MatrixFactorization-ML100K.ipynb` for starting with matrix factorization techniques on the MovieLens 100K dataset.

By following these steps, you'll be well-prepared to explore and learn from our TensorFlow-based recommendation systems, all tailored for an accessible, educational experience on CPU-only setups.

## Datasets

This project uses the MovieLens datasets in two versions to provide a comprehensive understanding of recommendation systems:

- **MovieLens 100K Dataset:** Contains 100,000 ratings from 1,000 users on 1,700 movies. It's stored in `data/MovieLens_100K/` and is perfect for initial experiments and learning purposes.
- **MovieLens 1M Dataset:** Comprises 1 million ratings from 6,000 users on 4,000 movies, located in `data/MovieLens_1M/`. This larger dataset allows for more in-depth analysis and model training.

These datasets have been chosen for their wide use in academia and industry, providing a solid foundation for exploring recommendation system techniques.

## Results

Some of our very simple experiments with the MovieLens datasets have yielded insightful results across different models:

| Model Category                | Dataset      | RMSE          | MAE           | MSE           | Precision@10 | Recall@10 | NDCG@10 |
|-------------------------------|--------------|---------------|---------------|---------------|--------------|-----------|---------|
| Matrix Factorization (LF=32)  | ML-100K      | 0.9352        | 0.7328        | 0.8746        | -            | -         | -       |
| Matrix Factorization (LF=32)  | ML-1M        | 0.9020        | 0.7056        | 0.8137        | -            | -         | -       |
| Item Autoencoder (Dim=500)    | ML-100K      | 1.4700        | 1.1304        | 2.1610        | 0.5397       | 0.7295    | 0.7922  |
| User Autoencoder (Dim=500)    | ML-100K      | 1.4523        | 1.1573        | 2.1091        | 0.4988       | 0.7053    | 0.8224  |

LF = Latent Factors, Dim = Encoding Dimension

## Future Plans

Looking ahead, we plan to:
- Integrate other neural architectures to deepen our exploration of recommendation systems.
- Experiment with additional datasets including Netflix, Amazon, Spotify, Filmtrust, and Douban Monti.
- Expand our collection of educative notebooks to cover more complex recommendation system problems.

### Key References

- **MovieLens Datasets**: F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context," in *ACM Transactions on Interactive Intelligent Systems*, vol. 5, no. 4, 2015. [DOI](https://doi.org/10.1145/2827872)
- **Matrix Factorization**: Y. Koren, R. Bell, and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," in *Computer*, Aug. 2009. [DOI](https://doi.ieeecomputersociety.org/10.1109/MC.2009.263)
- **AutoRec**: S. Sedhain et al., "AutoRec: Autoencoders Meet Collaborative Filtering," in Proceedings of the 24th International Conference on World Wide Web, 2015. [DOI](https://doi.org/10.1145/2740908.2742726)

## Acknowledgments

- Thanks to the GroupLens research group at the University of Minnesota for providing the MovieLens datasets.
- TensorFlow for the framework that enables building these models.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

*This repository was developed by Muhammad Junaid Ali Asif Raja.*
