using Microsoft.AspNetCore.Identity;

namespace Krcma.Eshop.Web.Models.Identity
{
    public class Role : IdentityRole<int>
    {
        public Role() : base() { }
        public Role(string role) : base(role) { }
    }
}
