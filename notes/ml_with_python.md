# Time series forecasting

1. define goal
    descriptive vs predictive
    1. purpose of generating forecasts
    2. type of forecasts that are needed
    3. how the forecasts will be used by the organization
    4. what are the costs associated with forecast errors
    5. what data will be available in the future
2. get data
3. explore & visualize
4. pre-process data
5. partition series
6. partition series
7. apply forecasting methods
8. evaluate & compare performance
9. implement forecasts/system

# Time series features

- date time features
    - important for future forecasts, can't use a date and then give it some random date in the future, needs to know day of week, weekend/not, week of year...
    - month, day of week, season...
- lag features
    - values at previous time steps
- window features
    - summary of values over a fixed window of prior time steps, avg sales of previous three months
    - rolling window: add a summary of the values at previous time steps (fixed window)
    - expanding windows: includes all previous data points in the series at each step

- upsampling and downsampling
    - upsampling: increase the frequency of the data
    - downlampling: decrease the frequency of the data

- power transformation and other transformations
    - log, sqrt...

- moving average smoothing

- exponential smoothing
    - assigning weights to previous averages as they contribute differently to current

- white noise
    - is a sequence of random numbers
    - if white noise no prediction
    - if no white noise, do prediction and test the error values, if error is white noise then good, if they are not white noise the model is still not extracting all information

- random walk 
    - drunkard walk
    - naive forcasting, next step is the current step. t+1 = t


time series model
y(t) = level + trend + seasonality + noise
or
y(t) = level * trend * seasonality * noise

- auto regression (AR)
    - no trend or seasonality


- moving average (MA)
    - 