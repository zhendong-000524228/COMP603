![image](https://github.com/zhendong-000524228/COMP603/assets/152204440/e3f1206e-1980-46a2-a8a4-294bfffa7d7c)# COMP603
Final Project of COMP603
Visualizations:
For the year 2022, create an appropriate graph(s) that shows the effect of COVID-19 on selected stock prices. Highlight on the graph if you notice any sharp change while comparing values. 
Consider the data range when you are comparing values in the graph. Explain why you chose the graph(s). You may use Matplotlib, Seaborn or Plotly library for graphs. [30 marks]

3 graphs, 1 general trends and 2 comparions between predicted trends and general trends.
1. General trends of both stock prices.
2. Comparison between Apple's general stock price trends and predicted trends(by the confirmed cases)
3. Comparison between Airbnb's general stock price trends and predicted trends(by the confirmed cases)
   
Create a horizontal bar chart of the top 20 countries according to the confirmed COVID-19 numbers for a given date. The bars should be sorted in ascending order. Use a colormap. [30 marks]

Ascending order in horizontal bar chart, I made a re-submission in Dropbox to make sure the graph is in ascending order.
top_20_countries = specific_date_data.sort_values('Confirmed', ascending=False).head(20)
The updated file will be uploaded here shortly after.
