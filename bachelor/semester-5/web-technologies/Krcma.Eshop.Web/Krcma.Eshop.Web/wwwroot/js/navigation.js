
class Section {
    constructor(pathname, element_link_id) {
        this.pathname = pathname;
        if (element_link_id == null) {
            this.element_link = null;
        } else {
            this.element_link = document.querySelector(element_link_id);
        }
    }

    pathNameConformity(pn) {
        var cnt = 0;
        for (var i = 0; i < pn.length && i < this.pathname.length; i++) {
            if (pn[i] == this.pathname[i]) {
                cnt++;
            }
        }
        return cnt / Math.max(pn.length, this.pathname.length);
    }
}

const sections = [
    new Section("/Home/Privacy", null),

    new Section("/Product/Shop", "#link_shop"),

    new Section("/ReturnModule/Manage", "#link_admin"),
    new Section("/Admin/Carousel", "#link_admin"),
    new Section("/Admin/Product", "#link_admin"),
    new Section("/Admin/Orders", "#link_admin"),
    new Section("/Admin/OrderItems", "#link_admin"),
    new Section("/Admin/Users", "#link_admin"),

    new Section("/Customer/CustomerOrders", "#link_my_orders"),

    new Section("/ReturnModule/Return", "#link_return"),

    new Section("/", "#link_home")
];

const pathname = window.location.pathname;

//najde sekci pro kterou je nejvetsi schoda pathname
var best = { value: -1, element_link: null};
sections.forEach((sec) => {
    var v = sec.pathNameConformity(pathname);
    if(v > best.value) {
        best.value = v;
        best.element_link = sec.element_link;
    }
});

if (best.element_link != null) {
    best.element_link.classList.add("active");
}