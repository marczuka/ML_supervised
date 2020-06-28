# ML_supervised
Implementing supervised machine learning models and resampling techniques to assess credit risk. We used the Logistic Regression model
with the standard parameters and different resampling techniques to address the class imbalance of the data (number of high risk loans 
is much lower than low risk loans).

We will analyse the obtained results to make a decision which of the the Logistic Regression ML algorythms (if any) should be used to assess the credit risks.

# Prediction results
<table>
  <tr>
    <td><h3>Random oversampling</h3>
    Balanced Accuracy Score = 0.648<br>
    Confusion matrix:
    <table>
      <tr>
        <th></th>
        <th>Predicted High</th>
        <th>Predicted Low</th>
      </tr>
      <tr>
        <th>True High</th>
        <td>70</td>
        <td>31</td>
      </tr>
      <tr>
        <th>True Low</th>
        <td>6789</td>
        <td>10315</td>
      </tr>
    </table>
    Classification imbalanced report:
    <table>
      <tr>
        <th></th>
        <th>pre</th>
        <th>rec</th>
        <th>f1</th>
        <th>sup</th>
      </tr>
      <tr>
        <th>High risk</th>
        <td>0.01</td>
        <td>0.69</td>
        <td>0.02</td>
        <td>101</td>
      </tr>
      <tr>
        <th>Low risk</th>
        <td>1.00</td>
        <td>0.60</td>
        <td>0.75</td>
        <td>17104</td>
      </tr>
    </table>
    </td>
    <td>
      <h3>SMOTE oversampling</h3>
      Balanced Accuracy Score = 0.662<br>
      Confusion matrix:
      <table>
        <tr>
          <th></th>
            <th>Predicted High</th>
            <th>Predicted Low</th>
          </tr>
          <tr>
            <th>True High</th>
            <td>64</td>
            <td>37</td>
          </tr>
          <tr>
            <th>True Low</th>
            <td>5277</td>
            <td>11827</td>
          </tr>
      </table>
      Classification imbalanced report:
      <table>
        <tr>
          <th></th>
          <th>pre</th>
          <th>rec</th>
          <th>f1</th>
          <th>sup</th>
        </tr>
        <tr>
          <th>High risk</th>
          <td>0.01</td>
          <td>0.63</td>
          <td>0.02</td>
          <td>101</td>
        </tr>
        <tr>
          <th>Low risk</th>
          <td>1.00</td>
          <td>0.69</td>
          <td>0.82</td>
          <td>17104</td>
        </tr>
      </table>
    </td>
  </tr>
  <tr>
    <td>
      <h3>ClusterCentroids Undersampling</h3>
      Balanced Accuracy Score = 0.533<br>
      Confusion matrix:
      <table>
        <tr>
          <th></th>
            <th>Predicted High</th>
            <th>Predicted Low</th>
          </tr>
          <tr>
            <th>True High</th>
            <td>67</td>
            <td>34</td>
          </tr>
          <tr>
            <th>True Low</th>
            <td>10217</td>
            <td>6887</td>
          </tr>
      </table>
      Classification imbalanced report:
      <table>
        <tr>
          <th></th>
          <th>pre</th>
          <th>rec</th>
          <th>f1</th>
          <th>sup</th>
        </tr>
        <tr>
          <th>High resk</th>
          <td>0.01</td>
          <td>0.66</td>
          <td>0.01</td>
          <td>101</td>
        </tr>
        <tr>
          <th>Low risk</th>
          <td>1.00</td>
          <td>0.40</td>
          <td>0.57</td>
          <td>17104</td>
        </tr>
      </table>
    </td>
    <td>
      <h3>SMOTEEN combined oversampling and undersampling</h3>
      Balanced Accuracy Score = 0.637<br>
      Confusion matrix:
      <table>
        <tr>
          <th></th>
          <th>Predicted High</th>
          <th>Predicted Low</th>
        </tr>
        <tr>
          <th>True High</th>
          <td>70</td>
          <td>31</td>
        </tr>
        <tr>
          <th>True Low</th>
          <td>7169</td>
          <td>9935</td>
        </tr>
      </table>
      Classification imbalanced report:
      <table>
        <tr>
          <th></th>
          <th>pre</th>
          <th>rec</th>
          <th>f1</th>
          <th>sup</th>
        </tr>
        <tr>
          <th>High risk</th>
          <td>0.01</td>
          <td>0.69</td>
          <td>0.02</td>
          <td>101</td>
        </tr>
        <tr>
          <th>Low risk</th>
          <td>1.00</td>
          <td>0.58</td>
          <td>0.73</td>
          <td>17104</td>
        </tr>
      </table>
    </td>
  </tr>
</table>
  
## Analysis of the results and conclusions
The main question while estimating Machine Learning models efficiency in credit risk assessments is to define what is more important to us: 
to give credit/loan to the most number of low risk loaners but sometimes miss the high risk loans (and give them a loan too) etc. to have 
a high precision, or to identify the maximum number of high risk loaners (to not risk that they won't be able to cover the loan) but also 
decline a lot of low risk loans due to high sensitivity of the model to high risk loans?

In all our attempts to resample the data our ML models precision for high risk loans is extremely small - only 1% - which means all the 
models are giving us a large number of false positives (etc. low risk loans classified as high risk). All models are giving us the best 
precision for low risk loans - 100% - so will will compare other parameters, like sensitivity.

We have two models with the highest sensitivity (among the others) to high risk loans: Random Oversampling and SMOTEEN. Those models were
able to predict 70 high risk loans out of 101 which is a quite good result (but not the best). However, looking at the sensitivity of those
models to the low risk loans, we have to admit that the number of false positives they return is too big - 6789 and 7169 respectively. And 
the numbers of true negatives (etc. the numbers of low risk loans also classified as low risk) are 10315 and 9935 respectively. This might 
suggest that we have to choose the Random Oversampling model since the numbers of true negatives and true positives are the highest for this
model, but I would say, that 10315 predicted low risk credits out of 17104 is still a not satisfying result.

As opposed to that, the SMOTE oversampling has the highest number of true negatives - 11827 predicted low risk loans out of 17104 actual low risks. 
This model's sensitivity to low risk loans is the highest among all the other models - 69%. While this model's sensitivity to the high risk loans 
is a bit lower than for the Random Oversampling and SMOTEEN models - 63% - I would recommend to still go with this model (if we are choosing between 
those four models only), since it's giving the most balanced result of true positives and true negatives. 
Balanced Accuracy Score for the SMOTE oversampling model is the highest among all four models - 66% - which also confirms the above conclusion.

Even though we were able to find "our winner" all those models show quite poor predicting results (all of them are lower than the 75th percentile).
According to that, I would strongly recommend to improve the LogisticRegression model by using, for example, another solver (liblinear shows the 
best results) or, at least, increasing the number of max_iter to 400-600.
