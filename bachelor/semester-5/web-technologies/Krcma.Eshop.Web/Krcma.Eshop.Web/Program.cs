using Krcma.Eshop.Web.Models.Database;
using Krcma.Eshop.Web.Models.Identity;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Identity;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web
{
    public class Program
    {
        public static void Main(string[] args)
        {
            IHost host = CreateHostBuilder(args).Build();

            using (var scope = host.Services.CreateScope())
            {
                if (scope.ServiceProvider.GetRequiredService<IWebHostEnvironment>().IsDevelopment())
                {
                    var dbContext = scope.ServiceProvider.GetRequiredService<EshopDbContext>();
                    DatabaseInit dbInit = new DatabaseInit();
                    dbInit.Initialization(dbContext);
                    var roleManager = scope.ServiceProvider.GetRequiredService<RoleManager<Role>>();
                    var userManager = scope.ServiceProvider.GetRequiredService<UserManager<User>>();
                    using (Task task = dbInit.EnsureRoleCreated(roleManager))
                    {
                        task.Wait();
                    }
                    using (Task task = dbInit.EnsureAdminCreated(userManager))
                    {
                        task.Wait();
                    }
                    using (Task task = dbInit.EnsureManagerCreated(userManager))
                    {
                        task.Wait();
                    }
                }
            }

            host.Run();
        }

        public static IHostBuilder CreateHostBuilder(string[] args) =>
          Host.CreateDefaultBuilder(args)
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.UseStartup<Startup>();
                })
                .ConfigureLogging((loggingBuilder) =>
                {
                    loggingBuilder.ClearProviders();
                    loggingBuilder.AddConsole();
                    loggingBuilder.AddDebug();
                    loggingBuilder.AddFile("Logs/eshop-log-{Date}.txt");
                });
    }
}
