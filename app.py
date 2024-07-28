import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt
import pickle
from datetime import date, datetime
import streamlit.components.v1 as components


from cryptocmd import CmcScraper
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


pickle_in = open("picklemodels/goldpickle", 'rb')
loadmodel = pickle.load(pickle_in)

pickle_in1 = open("picklemodels/silverpickle", 'rb')
loadmodel1 = pickle.load(pickle_in1)

def prediction(input_da):
    prediction = loadmodel.predict(input_da)
    return prediction

def prediction1(input_da):
    prediction1 = loadmodel1.predict(input_da)
    return prediction1

#components.html("<html><body><h1>CryptoCurrency Price Prediction</h1></html></body>")
st.title('CryptoVisionary')
st.markdown('Price Prediction Project')
st.markdown('All the prediction is done for Educational Purposes')


### Change sidebar color
st.markdown(
        """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#D6EAF8,#D6EAF8);
        color: black;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

### Set bigger font style
st.markdown(
        """
    <style>
    .big-font {
        fontWeight: bold;
        font-size:22px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title('CryptoVissionary')
options = st.sidebar.selectbox('Select a page:', 
                           ['Home','Cryptocurrency','Gold','Silver'])

if options == 'Home': # Prediction page
    st.markdown('Cryptocurrency, gold, and silver are three prominent assets that have garnered significant attention from investors, traders, and enthusiasts alike. Each asset possesses unique characteristics and dynamics that influence their prices, making them intriguing subjects for prediction and analysis. In this web application, users can access future price predictions for cryptocurrency, gold, and silver based on input dates, providing valuable insights for their investment decisions.')
    st.markdown('Cryptocurrency, the most recent addition to the financial landscape, has revolutionized traditional forms of currency with its decentralized nature and blockchain technology. Bitcoin, Ethereum, and numerous other altcoins have gained popularity and adoption over the years, attracting both retail and institutional investors. The volatile nature of cryptocurrencies presents both opportunities and risks for investors, making accurate price predictions crucial for informed decision-making.')
    st.markdown('Gold, often referred to as a "safe-haven" asset, has been valued for its rarity, durability, and intrinsic worth throughout history. Investors turn to gold during times of economic uncertainty or inflationary pressures, seeking stability and preservation of wealth. Price predictions for gold involve analyzing various factors such as geopolitical tensions, central bank policies, and demand-supply dynamics to anticipate future movements in its value.')
    st.markdown('Silver, often overshadowed by its more glamorous counterpart gold, is nevertheless a valuable precious metal with diverse industrial applications. Like gold, silver serves as a hedge against inflation and economic instability, attracting investors looking for portfolio diversification. Predicting silver prices involves assessing factors such as industrial demand, currency fluctuations, and macroeconomic trends to forecast its future trajectory accurately.')
    st.markdown('In this web application, users can input a specific date and receive predictions for the prices of cryptocurrency, gold, and silver for that future date. The prediction model employs advanced algorithms and machine learning techniques to analyze historical data, market trends, and relevant indicators to generate forecasts with a high degree of accuracy.')
    st.markdown('The web application interface is user-friendly, allowing individuals with varying levels of expertise to access and utilize the prediction tool effectively. Users can customize their input parameters, such as the date range and desired assets, to obtain tailored predictions that align with their investment objectives and risk tolerance.')
    st.markdown('Furthermore, the web application provides additional features such as interactive charts, real-time updates, and educational resources to enhance users understanding of cryptocurrency, gold, and silver markets. Through informative articles, tutorials, and expert insights, users can deepen their knowledge and make more informed investment decisions.')
    st.markdown('Overall, this web application serves as a valuable tool for investors seeking to navigate the dynamic landscape of cryptocurrency, gold, and silver markets. By leveraging advanced predictive analytics and comprehensive data analysis, users can gain valuable insights into future price movements and optimize their investment strategies accordingly. Whether seasoned traders or novice investors, individuals can benefit from the sophisticated forecasting capabilities offered by this innovative platform.')


elif options == 'Cryptocurrency': # Prediction page
    st.title('CryptoCurrency Price Predictor')
    def cyrto_app():

        ### Select ticker & number of days to predict on
        st.markdown("This application enables you to predict on the future value of any cryptocurrency (available on Coinmarketcap.com), for \
            any number of days into the future!")
        ### Select ticker & number of days to predict on
        cryptos= ("BTC","ETH","BNB","XRP","ADA","SOL","USDT","DOGE","DOT")
        selected_ticker = st.selectbox("Select crypto currency",cryptos)
        n_years= st.slider("Years of prediction",1,30)
        period = n_years * 365
        training_size = int(st.sidebar.number_input('Training set (%) size:', min_value=10, max_value=100, value=100, step=5)) / 100

        ### Initialise scraper without time interval
        def load_data(selected_ticker):
            init_scraper = CmcScraper(selected_ticker)
            df = init_scraper.get_dataframe()
            min_date = pd.to_datetime(min(df['Date']))
            max_date = pd.to_datetime(max(df['Date']))
            return min_date, max_date

        scraper = CmcScraper(selected_ticker)
        data = scraper.get_dataframe()
        
        st.subheader('Raw data')
        st.write(data.head())

        ### Plot functions
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        def plot_raw_data_log():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
            fig.update_yaxes(type="log")
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        ### Plot (log) data
        plot_log = st.checkbox("Plot log scale")
        if plot_log:
            plot_raw_data_log()
        else:
            plot_raw_data()

        ### Predict forecast with Prophet
        if st.button("Predict"):

            df_train = data[['Date','Close']]
            df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

            ### Create Prophet model
            m = Prophet(
                changepoint_range=training_size, # 0.8
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality=False,
                seasonality_mode='multiplicative', # multiplicative/additive
                changepoint_prior_scale=0.05
                )

            ### Add (additive) regressor
            for col in df_train.columns:
                if col not in ["ds", "y"]:
                    m.add_regressor(col, mode="additive")

            m.fit(df_train)

            ### Predict using the model
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            ### Show and plot forecast
            st.subheader('Predicted data')
            st.write(forecast.tail())

            st.subheader(f'Prediction plot for {period} days')
            fig1 = plot_plotly(m, forecast)
            if plot_log:
                fig1.update_yaxes(type="log")
            st.plotly_chart(fig1)

            st.subheader("Predicted components")
            fig2 = m.plot_components(forecast)
            st.write(fig2)

    if __name__ == "__main__":
        cyrto_app()

elif options == 'Gold':
    st.header('Gold Price Prediction')
    def goldmain():
        #st.write('''This data app uses Facebook's open-source Prophet library to automatically generate future forecast values from an imported dataset.
                    #You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast
                    #''')
        st.subheader("Select Date")
        # Add a date input widget
        date_input = st.date_input("Enter a date:", value=None, min_value=None, max_value=None)

        
        if st.button("Predict"):
            # Prepare the input data for prediction
            input_date = pd.to_datetime(date_input)
            input_data = pd.DataFrame({'ds': [input_date]})

            predictions = prediction(input_data)
            # Extract price, upper range, and lower range values
            price = predictions['yhat'].values[0]
            upper_range = predictions['yhat_upper'].values[0]
            lower_range = predictions['yhat_lower'].values[0]

            # Display the predictions
            st.subheader('Predicted Price:')
            st.write("Forecasted Gold Price($/gm):", price)
            st.write("Upper Range($/gm):", upper_range)
            st.write("Lower Range($/gm):", lower_range)
            # Generate future dates for prediction
            
        st.subheader("Select Date Range")    
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

    # Check if the start date is before the end date
        if start_date <= end_date:
        # Generate a range of dates between the start and end date
            date_range = pd.date_range(start=start_date, end=end_date)

        # Create a dataframe with the dates
            df = pd.DataFrame(date_range, columns=["ds"])

        else:
            st.error("Error: The start date must be before or equal to the end date.")    
        predictions = prediction(df)
        # Visualize the predicted graph
        st.subheader("Predicted Graph")
        chart_data = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        chart_data.set_index('ds', inplace=True)
        st.line_chart(chart_data)
        st.subheader("Table")
        st.write(predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    if __name__ == "__main__":
        goldmain()


elif options == 'Silver':
    st.header('Silver Price Prediction')
    def silvermain():
        #st.write('''This data app uses Facebook's open-source Prophet library to automatically generate future forecast values from an imported dataset.
                    #You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast
                    #''')
        st.subheader("Select Date")
        # Add a date input widget
        date_input = st.date_input("Enter a date:", value=None, min_value=None, max_value=None)

        
        if st.button("Predict"):
            # Prepare the input data for prediction
            input_date = pd.to_datetime(date_input)
            input_data = pd.DataFrame({'ds': [input_date]})

            predictions = prediction1(input_data)
            # Extract price, upper range, and lower range values
            price = predictions['yhat'].values[0]
            upper_range = predictions['yhat_upper'].values[0]
            lower_range = predictions['yhat_lower'].values[0]

            # Display the predictions
            st.subheader('Predicted Price:')
            st.write("Forecasted Silver Price($/gm):", price)
            st.write("Upper Range($/gm):", upper_range)
            st.write("Lower Range($/gm):", lower_range)
            # Generate future dates for prediction
            
        st.subheader("Select Date Range")    
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

    # Check if the start date is before the end date
        if start_date <= end_date:
        # Generate a range of dates between the start and end date
            date_range = pd.date_range(start=start_date, end=end_date)

        # Create a dataframe with the dates
            df = pd.DataFrame(date_range, columns=["ds"])

        else:
            st.error("Error: The start date must be before or equal to the end date.")    
        predictions = prediction1(df)
        # Visualize the predicted graph
        st.subheader("Predicted Graph")
        chart_data = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        chart_data.set_index('ds', inplace=True)
        st.line_chart(chart_data)
        st.subheader("Table")
        st.write(predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    if __name__ == "__main__":
        silvermain()