import os
import folium
import matplotlib.pyplot as plt
import numpy as np
import quandl
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import Form
from folium import FeatureGroup, LayerControl, Marker
from folium.plugins.beautify_icon import BeautifyIcon
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from wtforms.fields.html5 import DecimalRangeField
from wtforms import TextAreaField, validators
from wtforms.validators import NumberRange
from folium.plugins import HeatMap
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from nltk.corpus import wordnet

quandl.ApiConfig.api_key = "__vT6puk2ycpp-RXsgsY"

app = Flask(__name__)
Bootstrap(app)
app.config['SECRET_KEY'] = '1815'


class UserInputForm(Form):
    desc_txt = TextAreaField('Describe your ideal trip:',
                         validators=[validators.required(), validators.length(max=500)])

@app.route('/', methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
def index():
    map = EmptyMap(out_map='./templates/map.html')
    form = UserInputForm()
    if form.validate_on_submit():

        location_keyword = {'beach': ['beach', 'coast', 'surfing', 'cruise'],
                            'mall': ['mall', 'shopping', 'market'],
                            'restaurant': ['food', 'restaurant'],
                            'bar': ['bar', 'drink', 'nightlife'],
                            'park': ['park', 'fishing', 'hiking'],
                            'cultural': ['local', 'ethnic', 'music', 'historic', 'monument', 'museum', 'art'],
                            'hotel': [],
                            }
        mytext = form.desc_txt.data
        tag_weight = defaultdict(int)
        for key in location_keyword.keys():
            tag_weight[key] = 0
        sid = SentimentIntensityAnalyzer()
        for sen in sent_tokenize(mytext):
            ss = sid.polarity_scores(sen)
            for location_type in location_keyword.keys():
                for kw in location_keyword[location_type]:
                    if kw in sen:
                        tag_weight[location_type] += ss['compound']

        for location_type in location_keyword.keys():
            print(location_type, tag_weight[location_type])
        FindHotSpot(map, tag_weight=tag_weight, data_folder='./Data/', out_map='./templates/map.html')

    return render_template('index.html', form=form, out_map='/map')

@app.route('/map', methods=['GET'])
@app.route('/index.html', methods=['GET', 'POST'])
def show():
    return render_template('map.html')


def FindHotSpot(map, tag_weight, data_folder, out_map):
    count = 1
    pop = np.load(data_folder + 'DensityMaps/population.npy')[2]
    for tag in tag_weight.keys():
        kde_file = data_folder + 'DensityMaps/' + tag + '.npy'
        data = np.load(kde_file)
        kde = data[2]

        if count == 1:
            cum_kde = np.zeros_like(kde)

        # print(tag, tag_weight[tag])
        cum_kde += kde * float(tag_weight[tag]) / pop
        count += 1

    detected_peaks = detect_peaks(cum_kde)
    plt.subplot(2, 1, 1)
    plt.imshow(cum_kde)
    plt.subplot(2, 1, 2)
    plt.imshow(detected_peaks)

    x = data[0]
    y = data[1]
    peaks = np.array([x[detected_peaks == True],
                      y[detected_peaks == True],
                      kde[detected_peaks == True]])

    peaks = peaks[:, peaks[2, :].argsort()]
    feature_group = FeatureGroup(name='Suggested Places')
    for i in range(0, peaks.shape[1]):
        Marker(location=[peaks[0, i], peaks[1, i]], popup='Mt. Hood Meadows',
               icon=BeautifyIcon(border_color='#00ABDC', text_color='#00ABDC',
                                 number=(peaks.shape[1] - i), inner_icon_style='margin-top:0;')
               ).add_to(feature_group)

    feature_group.add_to(map)
    map = headmapmake(map, data_folder)
    LayerControl().add_to(map)
    map.save(out_map)


def EmptyMap(out_map):
    map = folium.Map([29, -82], zoom_start=6, tiles='stamentoner')
    map.save(out_map)
    return map


def detect_peaks(image):
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    background = (image == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    return detected_peaks


def headmapmake(map, data_folder):
    for file in os.listdir(data_folder + 'Points/'):
        point_file = data_folder + 'Points/' + file
        data = np.load(point_file)
        idx = np.random.randint(data.shape[1], size=int(data.shape[1]*0.1))
        data = data[:, idx]
        #print(np.transpose(data).shape)
        HeatMap(np.transpose(data).tolist(), name=file, show=False).add_to(map)
    return map


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, port=4433)
