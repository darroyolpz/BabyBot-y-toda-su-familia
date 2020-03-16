# Import modules
import schedule, warnings, json, requests, time, decimal, vlc, discord
from discord import Webhook, RequestsWebhookAdapter
from statistics import mean 
import pandas as pd
import numpy as np
import seaborn as sns
from functions_file import *
from pandas import ExcelWriter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

# Webhook settings
url_wb = os.environ.get('DISCORD_WH')
webhook = Webhook.from_url(url_wb, adapter=RequestsWebhookAdapter())

# Get the previous minutes closes ------------------------------------------------
def previous_closes(coin='BTC'):
	# Check execution time
	start_time = time.time()

	# Channel range
	channel_range = 0.004

	# Create dataframe
	df = coin_data_function(coin, start=datetime.now() + timedelta(days = -1), end = datetime.now(), tf='1m')

	# Sell volume
	df['USD sell volume'] = df['USD volume'].values - df['USD buy volume'].values

	# Buy-sell
	df['Buy-Sell'] = df['USD buy volume'].values - df['USD sell volume'].values

	# Gainz
	df['Gainz'] = 100*(df['Close'].values - df['Open'].values) / df['Open'].values

	# Z-function
	cols = ['USD buy volume', 'USD sell volume']

	for col in cols:
		col_name = 'Z-' + col
		df[col_name] = z_funct(df[col], 500)

	# Alarm
	btc_close = df['Close'].iloc[-1]
	gainz = df['Gainz'].iloc[-1]
	usd_sell = df['USD sell volume'].iloc[-1]
	z_sell = df['Z-USD sell volume'].iloc[-1]
	usd_buy = df['USD buy volume'].iloc[-1]
	z_buy = df['Z-USD buy volume'].iloc[-1]
	buy_sell = df['Buy-Sell'].iloc[-1]

	try:
		# First check the closes outside
		if ((z_sell > 3.8) and (gainz < 0) and (buy_sell < 0)) or ((z_buy > 3.8) and (gainz > 0) and (buy_sell > 0)):

			# Set the data to be checked and plotted
			df = df[-51:] # Only last 50 cases + break-out
			y = df['Close'].values
			length = len(y)
			x = np.arange(length, dtype=float)
			volume = df['USD sell volume'].values

			# Trend lines must be calculate with only one delay (which should be the breakout)
			y_trend = y[:-1]
			length_trend = len(y_trend)
			x_trend = np.arange(length_trend, dtype=float)
			y_trend = y_trend.reshape(length_trend, 1)
			regr = LinearRegression()
			regr.fit(x_trend.reshape(length_trend, 1), y_trend)
			y_predict = regr.predict(x_trend.reshape(length_trend, 1))
			#print('R2:', r2_score(y_trend, y_predict))

			# Trend points must be in the whole X range, not only in X_trend
			upper_range = y_predict*(1+channel_range)
			lower_range = y_predict*(1-channel_range)

			# Check if we have a clear range
			closes_outside_range = 0
			for row, ur, lr in zip(y, upper_range, lower_range):
				if (row < lr) or (row > ur):
					closes_outside_range += 1

			print('Closes outside range:', closes_outside_range)

			# No more than 6 closes outside the range - we need something smooth, not chop-suey
			if closes_outside_range < 7:

				# Scatter plot
				g = sns.scatterplot(x, y, size = volume, sizes=(5, 150), legend=False)
				g.set(xlim = (0, None))
				g.set(xlabel='Time sample')
				g.set_title(coin)

				# Trend lines plot
				plt.plot(x_trend, y_predict, '-.', color = 'green')
				plt.plot(x_trend, upper_range, '--', color = 'orange')
				plt.plot(x_trend, lower_range, '--', color = 'orange')

				# Save plot
				p_name = df['Open time'].iloc[-1]
				plot_name = 'BTC' + ' ' + str(datetime.now().strftime("%Y-%m-%d %H-%M")) + '.png'
				save_name = 'Charts/' + plot_name
				plt.savefig(save_name, format='png', dpi=300)

				# Short
				if buy_sell < 0:
					p = vlc.MediaPlayer("Short.mp3")
					p.play()
					sl, target = btc_close*(1+0.01), btc_close*(1-0.015)
					webhook.send(file=discord.File('mom.jpg'))
					webhook.send(f":chart_with_downwards_trend: Short the corn at {btc_close:.0f}\n:skull_crossbones: Stop loss: {sl:.0f}\n:dart: Target: {target:.0f}\n:scream_cat: USD sell volume: {usd_sell:.1f}\n:fire: Z-USD sell vol.: {z_sell:.1f}")
				# Long
				elif buy_sell > 0:
					p = vlc.MediaPlayer("Long.mp3")
					p.play()
					sl, target = btc_close*(1-0.01), btc_close*(1+0.015)
					webhook.send(file=discord.File('take_my_money.jpeg'))
					webhook.send(f":rocket: Send it from {btc_close:.0f}\n:skull_crossbones: Stop loss: {sl:.0f}\n:dart: Target: {target:.0f}\n:face_with_monocle: USD buy volume: {usd_buy:.1f}\n:fire: Z-USD buy vol.: {z_buy:.1f}")

				# Send the chart
				webhook.send(file=discord.File(save_name))

			# If more closes than desired
			else:
				p = vlc.MediaPlayer("Watchout.mp3")
				p.play()
				webhook.send(f":roller_coaster: Volatility! BTC price {btc_close:.0f}\n:x: Closes outside range: {closes_outside_range:.0f}\n:fire: Z-USD buy vol.: {z_buy:.1f}\n:scream_cat: Z-USD sell vol.: {z_sell:.1f}")

	except:
		p = vlc.MediaPlayer("Stop.mp3")
		p.play()
		print('Something failed motherfucker!')

	# Time stamp
	currentDT = datetime.now()
	print (str(currentDT))
	print('\n')

	# Screening
	cols = ['Close', 'Gainz','USD buy volume', 'USD sell volume', 'Z-USD buy volume', 'Z-USD sell volume']

	for col in cols:
		print(f"{col}: {float(df[col].iloc[-1]):.2f}")

	# Send to Excel
	#name ='BTC data.xlsx'
	#df.to_excel(name, index =  False)

	# Stop running time
	print('\n')
	print("--- %s seconds ---" % (time.time() - start_time))
	print('\n')

def rsi_job():
	# Only when internet is available
	df = coin_data_function('BTC', start=datetime(2017, 1, 1), end = datetime.now(), tf='1W')

	# RSI
	df['RSI'] = RSI(df['Close'])

	# Alarm
	limit = 28
	rsi_value = df['RSI'].iloc[-1]
	btc_close = df['Close'].iloc[-1]
	buy_price = btc_close*0.985
	sl = buy_price*0.90

	if rsi_value < limit:
		p = vlc.MediaPlayer("rsi_song.mp3")
		p.play()
		webhook.send(file=discord.File('take_my_money.jpeg'))
		webhook.send(f":dollar: BTC price: {btc_close}\n:fire: RSI value: {rsi_value:.2f}\n:money_mouth: Limit order (-1.5%) at {buy_price:.0f}\n:skull_crossbones: Stop loss (10%) at {sl:.0f}\n:dart: Target:   :waning_crescent_moon::dizzy:")

	currentDT = datetime.now()
	print (str(currentDT))
	print('BTC Price:', df['Close'].iloc[-1])
	print(f"RSI: {rsi_value:.2f}")
	print('\n')


# Start it just in case nobody is going to be hurt
rsi_job()
previous_closes()

# Schedule tasks
schedule.every().minute.at(":57").do(previous_closes)
schedule.every().minute.at(":30").do(rsi_job)
#schedule.every(15).seconds.do(previous_closes)
#schedule.every().day.at("01:01").do(you_only_had_one_job)
#schedule.every().minute.do(every_fucking_minute)

while True:
	schedule.run_pending()