# Workshop Recommender System

Welcome to the Workshop Recommender System repository. This project is designed to provide personalized workshop recommendations based on user preferences and historical data.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Source Data](#source-data)
- [Reference Article](#reference-article)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Workshop Recommender System leverages machine learning techniques to suggest workshops that align with user interests. By analyzing past interactions and preferences, the system aims to enhance user engagement and satisfaction.

## Features

- **Personalized Recommendations**: Tailors workshop suggestions based on individual user profiles.
- **Data-Driven Insights**: Utilizes historical data to improve recommendation accuracy.
- **Scalability**: Designed to handle a growing number of users and workshops efficiently.

## Source Data

The dataset used for this project is sourced from Kaggle:
[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

## Reference Article

This project is inspired by the following article:
[Recommender Systems in Python](https://www.datacamp.com/es/tutorial/recommender-systems-python)

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone git@github.com:iair/workshop_recommender_system.git
   cd workshop_recommender_system
   ```

2. **Install dependencies**:

   Ensure you have Python installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure API Key for OpenAI:
4. 
To use the recommendation system with AI capabilities, you will need to configure an API key for OpenAI.
Set up the key in your environment variables:

    ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
The system is configured to use the 40-mini model for recommendatio


## Usage

Once installed, you can start executing the notebooks to check the differents results

## Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Please ensure your code adheres to our coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

