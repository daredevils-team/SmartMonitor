using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SmartMonitor.Models
{
    public class PieceOfNews
    {
        public string title { get; set; }
        public string author { get; set; }
        public string description { get; set; }
        public string url { get; set; }
        public DateTime? publishedAt { get; set; }
    }
}
