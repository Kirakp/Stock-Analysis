import pandas as pd
import plotly.graph_objects as go
import math
import dash
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import dash_core_components as dcc
import dash_html_components as html
import base64
# ----------------------------------------------
app = dash.Dash(__name__)
# ----------------------------------------------
df = pd.read_csv('NFLX.csv')
df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']
# ----------------------------------------------
fig = go.Figure(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']))
# ----------------------------------------------
plt.figure(figsize=(15,5))
plt.title('Closing Price History')
plt.plot(df['Close'])
plt.xlabel('--Date--',fontsize=10)
plt.ylabel('--Closing Price in USD--',fontsize=10)
plt.savefig(format('pi'))
fi='pi.png'
encode_image = base64.b64encode(open(fi, 'rb').read())
#-----------------------------------------------
data=df.filter(['Close'])
dataset=data.values
training_data_len=math.ceil(len(dataset)*.8)
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
train_data=scaled_data[0:training_data_len,:]
x_train,y_train=[],[]
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1,epochs=1,verbose=2)
test_data=scaled_data[training_data_len-60:,:]
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)
rmse=np.sqrt(np.mean(predictions-y_test)**2)
train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions
# ----------------------------------------------
plt.figure(figsize=(15,5))
plt.title('Model')
plt.xlabel('--Date--', fontsize=10)
plt.ylabel('--Close Price in USD--', fontsize=10)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Value', 'Predictions'], loc='upper right')
plt.savefig(format('pic'))
fig1='pic.png'
encoded_image = base64.b64encode(open(fig1, 'rb').read())
# ----------------------------------------------
app.layout = html.Div(children=[
    html.H1(children='Stock Analysis',
            style={
                'textAlign' : 'center',
                'color' : '#2E4C6D'
            }),
    html.H2(children='''
        Dashboard: Showing the analysis of the stock using LSTM.
    ''', style={
                'textAlign' : 'center',
                'color' : '#396EB0'
    }),
    html.H3(
        children='Initial: Stock Value in Candle stick chart',
        style={
                'textAlign' : 'center',
                'color' : '#95D1CC'}
    ),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    html.H3(
        children='Actual: Stock Value in Line chart',
        style={
                'textAlign' : 'center',
                'color' : '#95D1CC'}
    ),
    html.Img(src='data:image/png;base64,{}'.format(encode_image.decode()),style={'Align' : 'center'}),
    html.H3(
        children='Prediction: Stock Analysis',
        style={
                'textAlign' : 'center',
                'color' : '#95D1CC'}
    ),
    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),style={'Align' : 'center'})
])
# -------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)