using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Reflection;
using System.Threading.Tasks;
using Blazorise;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using NewsAPI;
using NewsAPI.Constants;
using NewsAPI.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SmartMonitor.Models;

namespace SmartMonitor.Controllers
{

    public class DetailsController : Controller
    {
        public IActionResult Index(string id)
        {
            var News = new NewsViewModel();
            var dir = @"/images/" + id.Replace(" ", "_") + @"/" + id.Replace(" ", "_");
            News.img1 = dir + "1.png";
            News.img2 = dir + "2.png";
            News.img3 = dir + "3.png";
            News.img4 = dir + "4.png";
            News.dt = Convert.ToDateTime(System.IO.File.ReadAllText(Environment.CurrentDirectory + @"\wwwroot\images\" + id.Replace(" ", "_") + @"\" + "date.txt"));
            News.coeff = System.IO.File.ReadAllText(Environment.CurrentDirectory + @"\wwwroot\images\" + id.Replace(" ", "_") + @"\" + "res.txt");
            var news = new List<PieceOfNews>();
            var newsApiClient = new NewsApiClient("c4cdf6d372ae4e6badf36a0851ba882c");
            var articlesResponse = newsApiClient.GetEverything(new EverythingRequest
            {
                Q = id.Replace(" ","+"),
                SortBy = SortBys.Relevancy,
                Language = Languages.EN,
                PageSize = 5,
            });
            if (articlesResponse.Status == Statuses.Ok)
            {
                foreach (var article in articlesResponse.Articles)
                {
                    var currentPoN = new PieceOfNews {
                        title = article.Title,
                        author = article.Author,
                        description = article.Description,
                        url = article.Url,
                        publishedAt = article.PublishedAt,
                    };
                    news.Add(currentPoN);
                }
            }
            News.News = news;
            
            return View(News);
        }

    }


}
