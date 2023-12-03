from flask import Flask, request, render_template
import settings
import utils
import numpy as np
import json
import cv2
import predictions as pred


app = Flask(__name__)
app.secret_key = 'document_scanning app'

docscan = utils.DocumentScan()


@app.route('/', methods = ['GET', 'POST'])
def scan_document():
    
    if request.method == 'POST':
        file = request.files['image_name']
        upload_img_path = utils.save_upload_image(file)
        print('Image saved in : ', upload_img_path)
        
        #predict co-ordinates
        four_points, size = docscan.document_scanner(upload_img_path)
        print(f'Four Points : {four_points}, Image Size : {size}')
        if four_points is None:
            
            message = 'Unable to locate co-ordinates of points. Random points displayed'
            #points to be given to JS. Need to be given in JSON format
            points = [{ 'x' : 0, 'y': 0}, {'x' : 10, 'y': 4}, 
                      { 'x': 120, 'y': 150}, { 'x' : 40, 'y': 20} ]
            
            return render_template('scanner.html', points = points, fileupload = True, message = message)
        
        else:
            points = utils.four_points_json(four_points)
            message = 'Located the co-ordinates using opencv'
            return render_template('scanner.html', points = points, fileupload = True, message = message)
            
    return render_template('scanner.html')

@app.route('/transform', methods = ['POST'])
def transform():
    
    try:
        points = request.json['data']
        array = np.array(points)
        img = docscan.caliberate_to_original_size(array)
        filename = 'magic_color_img.jpg'
        magic_image_path = settings.join_path(settings.MEDIA_DIR, filename)
        cv2.imwrite(magic_image_path, img)
        return 'success'
    
    except:
        return 'failure'

@app.route('/prediction')
def predictions():
    image = cv2.imread(settings.join_path(settings.MEDIA_DIR, 'magic_color_img.jpg'))
    img_bb, results = pred.prediction(image)
    cv2.imwrite(settings.join_path(settings.MEDIA_DIR, 'bounding_box.jpg'), img_bb)
    return render_template('predictions.html', results = results)


@app.route('/about')
def about():
    
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug = True)