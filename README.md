# JunctionX Seoul 2020

# SmartMonitor

Detect damaged building by natural disasters and get a brief information about it.

### Technologies used
For development of the web app we used following technologies:
- ASP.NET Core;
- [Bulma](https://bulma.io);
- [NewsAPI](https://newsapi.org);

Machine learning side was developed with **PyTorch**. We used Res-Net50 pretrained on COCO dataset.

### How it works
1. Pick up a natural disaster;
2. Check news and photos with detected by ResNet damaged houses

## Presentation
Check on [Google Docs](https://docs.google.com/presentation/d/e/2PACX-1vQGgs65mox96CRPLiuKG7pkToq_3VL4xF8cz6vKprEPQI5A4dg9TZyJkIb6WLY3hIrAt9Pazc-4pzoj/pub)

### Future plans
Due to the lack of experience we haven't managed to successfully integrate it with Azure and deploy it but we're looking forward to improve our project!
Our future plans are:
- [ ] Azure integration;
- [ ] Adding data storage;
- [ ] Better natural disaster news aggregation;
- [ ] Retraining neural network to be able to detect other objects and show more detailed information about damage;
- [ ] Azure/DigiatOcean Deployment
