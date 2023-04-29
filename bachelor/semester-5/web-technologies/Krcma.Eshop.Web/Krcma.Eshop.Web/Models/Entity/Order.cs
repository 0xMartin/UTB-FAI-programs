using Krcma.Eshop.Web.Models.Identity;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Krcma.Eshop.Web.Models.Entity
{
    [Table(nameof(Order))]
    public class Order
    {

        [Key]
        [Required]
        public int ID { get; set; }

        [DatabaseGenerated(DatabaseGeneratedOption.Computed)]
        public DateTime DateTimeCreated { get; protected set; }

        public int ShippingDelay { get; set; }

        [StringLength(25)]
        [Required]
        public string OrderNumber { get; set; }

        [Required]
        public double TotalPrice { get; set; }

        [ForeignKey(nameof(User))]
        public int UserId { get; set; }
        public User User { get; set; }

        //[ForeignKey(nameof(OrderStatus))]
        //public int OrderStatusId { get; set; }
        //public OrderStatus OrderStatus { get; set; }

        public IList<OrderItem> OrderItems { get; set; }

    }
}
