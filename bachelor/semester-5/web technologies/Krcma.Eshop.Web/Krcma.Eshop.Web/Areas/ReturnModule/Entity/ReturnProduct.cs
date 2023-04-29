using Krcma.Eshop.Web.Models.Entity;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Threading.Tasks;

namespace Krcma.Eshop.Web.Areas.ReturnModule.Entity
{

    [Table(nameof(ReturnProduct))]
    public class ReturnProduct
    {

        [Key]
        [Required]
        public int ID { get; set; }

        [ForeignKey(nameof(OrderItem))]
        public int OrderItemID { get; set; }

        public OrderItem OrderItem { get; set; }

        [Required]
        public bool Approved { get; set; }

        [Required]
        public bool Processed { get; set; }

        [NotMapped]
        public int RemainingDays { get; set; }
    }
}
