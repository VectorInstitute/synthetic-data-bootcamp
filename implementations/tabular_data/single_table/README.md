## TabDDPM: modelling tabular data with diffusion models

### Background: Diffusion models
Every diffusion model (image, audio, or tabular) is built on one trick: take real data, destroy it with noise in a slow, controlled way, then train a network to undo that destruction one small step at a time.
The same recipe works on a row of a spreadsheet instead of a grid of pixels in an image. We just need to define what "adding noise" and "removing noise" mean for spreadsheet-shaped data.

#### The difference: 
A spreadsheet row is not one homogeneous thing. A pixel is always a continuous brightness value, and neighboring pixels are spatially related. A table row is a grab-bag: age is a continuous number, city is a category with no natural ordering, has_subscription is binary. We can't just "add Gaussian noise" to a category the way you can to a number (pixel in an image)

### TabDDPM

They combine gaussian diffusion and multinomial diffusion to model numerical and categorical features, respectively. Gaussian diffusion models operate in continuous spaces where forward and reverse processes are characterized by Gaussian distributions, while multinomial diffusion are designed to generate categorical data and employs categorical distribution.


<div align="center">
  <img src="./images/tabddpm.png" alt="TabDDPM" width="600" height="280">
</div>




Source: https://research.yandex.com/blog/tabddpm-modelling-tabular-data-with-diffusion-models

## Example dataset: Berka
The Berka dataset is a collection of financial information from a Czech bank. The dataset deals with over 5,300 bank clients with approximately 1,000,000 transactions. Additionally, the bank represented in the dataset has extended close to 700 loans and issued nearly 900 credit cards, all of which are represented in the data."
Download link: https://www.kaggle.com/datasets/marceloventura/the-berka-dataset

In the single-table reference implementation, we work with the transaction table of this dataset. 
- `trans.csv` table information: https://webpages.charlotte.edu/mirsad/itcs6265/group1/transaction_domain.html

