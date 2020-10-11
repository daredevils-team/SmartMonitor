using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SmartMonitor.Models
{
    public class Event
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public DateTime Beginning { get; set; }
        public DateTime Ending { get; set; }
        public byte[] Img1 { get; set; }
        public byte[] Img2 { get; set; }
        public byte[] Img3 { get; set; }
        public byte[] Img4 { get; set; }
        public int numberBefore { get; set; }
        public int numberAfter { get; set; }
    }
}
