# GENERATORLARI EGITEMDEKI GIBI GUNCELLE



from keras.models import Model,Sequential,load_model
from keras.utils.vis_utils import plot_model
import pandas as pd
from utils import RgbGenerator
from models import ModelFactory,ModelType
train=pd.read_csv('20bnjester_csv_files/train.csv')
valid=pd.read_csv('20bnjester_csv_files/valid.csv')


model =ModelFactory(rgbpath='trained_models/rgblstm.h5',trained=True).getModel(ModelType.RGB)
model.summary()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

predict_gen = RgbGenerator(valid)

y_pred = model.predict_generator(predict_gen,1750)


y_pred = np.argmax(y_pred,axis=1)

print(y_pred)

y_true = np.argmax(valid.iloc[:14000,1:].values,axis=1)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt     
cm = confusion_matrix(y_true,y_pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, xticklabels=True, yticklabels=True); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(list(valid)[1:]); ax.yaxis.set_ticklabels(list(valid)[1:],rotation =0);
plt.show()
print(classification_report(y_true, y_pred, target_names=list(valid)[1:]))


