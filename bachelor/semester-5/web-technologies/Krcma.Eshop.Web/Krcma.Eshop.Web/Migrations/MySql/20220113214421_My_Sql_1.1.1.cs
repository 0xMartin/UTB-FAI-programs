using Microsoft.EntityFrameworkCore.Migrations;

namespace Krcma.Eshop.Web.Migrations.MySql
{
    public partial class My_Sql_111 : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "Processed",
                table: "ReturnProduct",
                type: "tinyint(1)",
                nullable: false,
                defaultValue: false);
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Processed",
                table: "ReturnProduct");
        }
    }
}
