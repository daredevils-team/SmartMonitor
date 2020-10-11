using Microsoft.AspNetCore.Identity;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SmartMonitor.Models
{
    public class NewsViewModel
    {
        public List<PieceOfNews> News { get; set; }
        public DateTime dt { get; set; }
        public string img1 { get; set; }
        public string img2 { get; set; }
        public string img3 { get; set; }
        public string img4 { get; set; }
        public string coeff { get; set; }
    }
}
