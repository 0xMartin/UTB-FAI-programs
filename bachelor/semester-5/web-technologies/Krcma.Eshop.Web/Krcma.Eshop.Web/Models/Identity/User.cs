using Microsoft.AspNetCore.Identity;

namespace Krcma.Eshop.Web.Models.Identity
{
    public class User : IdentityUser<int>
    {
        public string FirstName { get; set; }
        public string LastName { get; set; }
    }
}
