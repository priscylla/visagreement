# Visagreement

Visagreement is a web-based tool designed for exploring the (dis)agreement among various machine learning model explanation methods. As the use of machine learning models becomes increasingly prevalent, understanding and assessing the reliability of explanation methods are crucial. Visagreement empowers users to dive into the complexities of these explanations.

```python
from visagreement import Visagreement

visagreement_tool = Visagreement('name')

visagreement_tool.load_model(model, X_train, X_test, y_test)
```
```python
from captum.attr import ( KernelShap, Lime, DeepLift, IntegratedGradients)

visagreement_tool.create_explanations('KS', KernelShap)
visagreement_tool.create_explanations('Lime', Lime)
visagreement_tool.create_explanations('DL', DeepLift)
visagreement_tool.create_explanations('IG', IntegratedGradients)
```

```python
python app.py
```
