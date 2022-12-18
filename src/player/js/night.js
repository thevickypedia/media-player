var theme = window.localStorage.currentTheme;
$('body').addClass(theme);
if ($("body").hasClass("night")) {
    $('.toggler').addClass('fa-sun-o');
    $('.toggler').removeClass('fa-moon-o');
} else {
    $('.toggler').removeClass('fa-sun-o');
    $('.toggler').addClass('fa-moon-o');
}
$('.toggler').click(function() {
    $('.toggler').toggleClass('fa-sun-o');
    $('.toggler').toggleClass('fa-moon-o');
    if ($("body").hasClass("night")) {
        $('body').toggleClass('night');
        localStorage.removeItem('currentTheme');
        localStorage.currentTheme = "day";
    } else {
        $('body').toggleClass('night');
        localStorage.removeItem('currentTheme');
        localStorage.currentTheme = "night";
    }
});
