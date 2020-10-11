using Microsoft.EntityFrameworkCore;

namespace SmartMonitor.Models
{
    public class MyDbContext : DbContext
    {
        public virtual System.Data.Entity.DbSet<Event> Events { get; set; }

        protected override void OnConfiguring(DbContextOptionsBuilder options)
            => options.UseSqlite("Data Source=SmartMonitor.db");
    }
}
