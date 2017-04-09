# Handwriting Recognition
### Arda Mavi - [ardamavi.com](http://www.ardamavi.com/)

Handwriting recognition with machine learning.

I used decision trees from [Scikit-Learn](http://scikit-learn.org) for finding which character in image.
Scikit-Learn is an open source Python library for machine learning.

### Using Commend: <br/>
`python3 main.py <ImageFileName>`

### Adding new characters and train dataset:
If you want to adding new character to datasets, you create a directory and rename what you want to add character (like 'a' or 'i').

If you want to add a new training images to previously character datasets or null dataset of character, you add a image(suggested image size: 25x25) to about character directory.

Note: We work on grayscale(2D matrix) image also if you use color images(like RGB), program automatically return to grayscale from your color image.

### Used Modules:
- sys
- numpy
- PIL (Image)
- os (listdir)
- sklearn (joblib and MLPClassifier)

### Important Notes:
- Suggested image size: 25x25
- Install above modules
- If you want to update classifier(maybe for newly trained classifier with new data sets), delete 'Data/classifier/classifier.pkl' file and run program again.
- The program can only distinguish  '0', '1' and 'A' characters. To be able to distinguish other characters, you should add new character datasets.
