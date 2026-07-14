# Spatio-Temporal Prediction and Coordination of EV Charging Demand for Power System Resilience

## Interactive demo

**▶ [richelcode.github.io/ev-charging-demand-demo](https://richelcode.github.io/ev-charging-demand-demo/)**

Explore this project's results in the browser: GraphWaveNet-GRU-LSTM forecasts against a Random
Forest baseline across 6 Caltrans PeMS District 3 stations and 12/24/48/72 h horizons, the EV
charging load derived from those forecasts, and how each model holds up as sensors go dark.
Under 30% sensor outage the baseline's error grows about 2.8x faster than the graph model's.

[![Interactive dashboard](https://raw.githubusercontent.com/RichelCode/ev-charging-demand-demo/main/screenshot.png)](https://richelcode.github.io/ev-charging-demand-demo/)

Source for the demo: [RichelCode/ev-charging-demand-demo](https://github.com/RichelCode/ev-charging-demand-demo)

## Research Objectives

Recent studies have explored electric vehicles (EVs) from different perspectives, ranging from estimating vehicle range based on battery capacity, model specifications, and internal components (Ahmed et al., 2022) to forecasting charging behavior using machine learning methods such as Random Forest and SVM with factors like previous payment data, weather, and traffic (Shahriar et al., 2020). In parallel, research on smart cities has focused on managing traffic flow efficiently to reduce congestion and energy consumption (Dymora, Mazurek, & Jucha, 2024).

Building on these insights, this project connects traffic dynamics with EV energy consumption to better predict when and where charging demand will arise. By integrating spatio-temporal traffic features with deep learning models, the objective is to anticipate EV charging needs in real time and support coordinated charging strategies that enhance overall power system resilience.

The goal is to create a data-driven framework that uses traffic flow, speed, and spatial patterns to model charging demand more accurately. This forms the foundation for developing robust, scalable prediction systems that help improve grid stability and advance the integration of electric vehicles in smart cities.
