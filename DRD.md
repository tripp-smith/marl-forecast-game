# Data Requirements Document for Multi-Agent Adversarial Forecasting Framework

## 1. Purpose
This document outlines open-source intelligence (OSINT) sources required for the framework to support robust forecasting in dynamic environments. Sources include quantitative data series (e.g., time-series metrics, forecasts) and qualitative outlooks (e.g., expert commentary, market analyses). These enable agent modeling of macroeconomic indicators, sector-specific trends, and adversarial disturbances. Sources are selected for public accessibility, reliability, and relevance to generic forecasting applications such as demand prediction or economic modeling. Data ingestion should prioritize APIs where available for real-time integration, with fallbacks to web scraping or manual downloads.

## 2. Data Categories and Sources
Sources are categorized by type, with descriptions of content, format, access methods, and relevance to the framework (e.g., for state modeling in Markov games or AML defenses).

### 2.1 Prediction Markets
These provide crowd-sourced probabilistic forecasts on economic events, useful for adversarial simulations and qualitative outlooks on market sentiment.

- **Polymarket**  
  Description: Decentralized prediction market platform offering contracts on macro indicators (e.g., GDP growth, inflation rates), geopolitical events, and sector outcomes (e.g., tech stock performance). Includes real-time odds and historical resolutions for quantitative benchmarking.  
  Data Types: Quantitative (probabilities, trading volumes); Qualitative (event commentaries).  
  Access: API via polymarket.com/api; Web interface at polymarket.com/predictions/macro-indicators.  
  Relevance: Simulates adversarial economic trade-offs; integrates with agents for Nash equilibrium calculations.

- **Kalshi**  
  Description: CFTC-regulated prediction market for economic indicators (e.g., unemployment rates, Fed decisions) and sector events (e.g., energy prices). Provides time-series odds and resolution data.  
  Data Types: Quantitative (market prices, volumes); Qualitative (event descriptions).  
  Access: API at kalshi.com/api; Web at kalshi.com/markets.  
  Relevance: Enhances robustness testing against disturbances like policy shifts.

- **PredictIt**  
  Description: Academic-oriented market for political and economic forecasts (e.g., election impacts on markets). Offers historical data series on resolved contracts.  
  Data Types: Quantitative (yes/no probabilities); Qualitative (user discussions).  
  Access: API via predictit.org/api; Web at predictit.org.  
  Relevance: Useful for qualitative geopolitical outlooks affecting macro forecasts.

### 2.2 Macroeconomic Data Series and Forecasts
Authoritative global and national sources for quantitative macro indicators, including time-series data and forward-looking projections.

- **Federal Reserve Economic Data (FRED)**  
  Description: Comprehensive database of U.S. and global economic indicators (e.g., GDP, inflation, interest rates) with historical series and forecasts.  
  Data Types: Quantitative (time-series, APIs for real-time updates); Qualitative (research papers).  
  Access: API at fred.stlouisfed.org/docs/api; Web at fred.stlouisfed.org.  
  Relevance: Core for state transitions in Markov games; supports AML training on economic shifts.

- **International Monetary Fund (IMF) Data**  
  Description: Global economic outlooks, including World Economic Outlook (WEO) forecasts for GDP, inflation, and trade balances.  
  Data Types: Quantitative (datasets, projections); Qualitative (analytical reports).  
  Access: API at data.imf.org; Web at imf.org/en/Data.  
  Relevance: Provides macro baselines for adversarial scenario modeling.

- **World Bank Open Data**  
  Description: Indicators on development, poverty, and economic growth (e.g., sectoral GDP contributions) with forecasts.  
  Data Types: Quantitative (time-series, APIs); Qualitative (country reports).  
  Access: API at data.worldbank.org; Web at data.worldbank.org.  
  Relevance: Enables cross-country comparisons for multinational forecasting.

- **Organisation for Economic Co-operation and Development (OECD) Data**  
  Description: Economic forecasts and indicators (e.g., productivity, unemployment) for member countries.  
  Data Types: Quantitative (forecast databases); Qualitative (policy analyses).  
  Access: API at data.oecd.org; Web at data.oecd.org.  
  Relevance: Supports sector-specific macro linkages, e.g., tech industry growth projections.

- **Bureau of Economic Analysis (BEA)**  
  Description: U.S. GDP, trade, and industry accounts with quarterly forecasts.  
  Data Types: Quantitative (detailed series); Qualitative (economic accounts commentary).  
  Access: API at bea.gov/api; Web at bea.gov.  
  Relevance: Quantitative inputs for agent reward functions.

### 2.3 Sector-Specific Authoritative References
Tailorable to domains like semiconductors, finance, or energy; examples provided for adaptability.

- **Semiconductor Industry Association (SIA)** (Sector: Semiconductors)  
  Description: Global semiconductor sales data, forecasts, and market outlooks (e.g., World Semiconductor Trade Statistics).  
  Data Types: Quantitative (monthly sales series); Qualitative (industry reports).  
  Access: Web at semiconduct.org/resources; API-limited.  
  Relevance: Sector demand modeling; adversarial disturbances like supply chain risks.

- **BloombergNEF** (Sector: Energy/Tech)  
  Description: Forecasts on renewable energy, EVs, and tech adoption impacting sectors like semiconductors.  
  Data Types: Quantitative (projections); Qualitative (long-term outlooks).  
  Access: Partial open access at bnef.com; Reports downloadable.  
  Relevance: Qualitative commentary for agent refactoring.

- **National Bureau of Economic Research (NBER)** (Cross-Sector)  
  Description: Working papers and data on business cycles, with sector-specific analyses (e.g., tech innovation impacts).  
  Data Types: Quantitative (datasets); Qualitative (research commentaries).  
  Access: Web at nber.org/data; Free downloads.  
  Relevance: Historical data for time-series forecasting in agents.

- **Eurostat** (Sector: EU Markets)  
  Description: Sectoral statistics (e.g., manufacturing output, trade) with forecasts.  
  Data Types: Quantitative (time-series); Qualitative (regional reports).  
  Access: API at ec.europa.eu/eurostat/api; Web at ec.europa.eu/eurostat.  
  Relevance: Regional macro-sector integration.

### 2.4 Qualitative Outlooks and Commentary Sources
Publicly available analyses for contextualizing quantitative data.

- **Reuters Economic News**  
  Description: Real-time commentary on macro events and sector trends (e.g., supply chain disruptions).  
  Data Types: Qualitative (articles, analyses).  
  Access: Web at reuters.com/business/economy; RSS feeds.  
  Relevance: Inputs for adversary agents simulating market hacks.

- **Brookings Institution**  
  Description: Policy briefs and economic outlooks (e.g., global economy commentaries).  
  Data Types: Qualitative (reports, blogs).  
  Access: Web at brookings.edu/topic/economics.  
  Relevance: Expert insights for LLM-driven refactoring.

- **Social Media and Forums** (e.g., Reddit r/economics, X)  
  Description: Community discussions on forecasts and events.  
  Data Types: Qualitative (threads, posts).  
  Access: APIs for X/Reddit.  
  Relevance: Real-time sentiment for adversarial modeling.

## 3. Data Integration Requirements
- **Frequency**: Real-time for markets; Quarterly/annual for forecasts.  
- **Formats**: CSV/JSON for quantitative; Text/PDF for qualitative.  
- **Validation**: Cross-reference sources for robustness; Use AML checks for data poisoning.  
- **Storage**: Persistent database (e.g., SQLite) for shared agent access.  
- **Ethical Considerations**: Ensure compliance with OSINT ethics; Avoid non-public data.  

This list is extensible; prioritize automation for framework scalability.