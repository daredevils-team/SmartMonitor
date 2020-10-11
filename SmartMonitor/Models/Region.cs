using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SmartMonitor.Models
{
    public class Region
    {
        public CurrentRegion CurrentRegion { get; set; }
    } 

    public enum CurrentRegion
    {
        Italy, Russia
    }
}
