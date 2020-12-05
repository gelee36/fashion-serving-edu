# Fashion Recommendations

A standalone HTTP web server that can recommend similar fashion outfits.

Uses multiple neural networks (with a ResNet50 backbone) behind the scenes to classify inputs by {category, texture, fabric, parts, shape}. The resulting embeddings are then used to query a pre-built nearest neighbors index for similar outputs.


## Installation

Use [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt
```

## Usage

To run the web server, simply execute flask with the main recommender app:

```sh
flask run
```

The main predictor can also be used independently of Flask, by calling `get_recs`:

```python
from predict import Predict

fashion = Predict()
recs = fashion.get_recs(img_path)
```

## Built With

* [fast.ai](https://www.fast.ai/) - Deep learning library used for CNN training
* [Flask](http://flask.pocoo.org/) - Python HTTP server

## Files
* [Outfits](https://github.com/sds-arch-cert/Fasion_serving_edu/tree/main/Outfits) - Example images that can be used to test the recommendation system 
* [app.py](https://github.com/sds-arch-cert/Fasion_serving_edu/blob/main/app.py) - Spins up a Flask App to serve recommendations 
* [predict.py](https://github.com/sds-arch-cert/Fasion_serving_edu/blob/main/predict.py) - Recommendation System

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
