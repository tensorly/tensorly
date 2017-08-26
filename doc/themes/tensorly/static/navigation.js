 // Top bar burger
(function() {
var burger = document.querySelector('.navbar-burger');
var target_name = burger.dataset.target;
var target = document.getElementById(target_name);
    burger.addEventListener('click', function() {
        burger.classList.toggle('is-active');
        target.classList.toggle('is-active');
    });
})();


// SIDEBAR
// Fix sidebar on scroll
sidebar_offset = document.getElementById('sidebar').offsetTop;
var timer=0;
window.addEventListener('scroll', function(e) {
    //if(timer) { window.clearTimeout(timer); }
    document.getElementById('sidebar').classList[window.scrollY > sidebar_offset ? 'add' : 'remove']('stick');
    timer = window.setTimeout(function() {
    }, 10);
});

// make current position active on the sidebar
function nohl_current(element) {
    if (typeof element != 'undefined') {
        element.parentNode.classList.remove('current');
        element.parentNode.classList.remove('show');
    }
}
function highlight_current(element) {
    if (typeof element != 'undefined') {
        if (!element.classList.contains('current')) {
            element.parentNode.classList.add('current');
        }
    }
}

var position = window.location.hash;
var current_element = document.querySelectorAll('a[href="'+position+'"]')[0];
var new_element = current_element;
highlight_current(current_element);

// toggle show 
var toggle_list_show = function() {
    // var attribute = this.parentNode.querySelector("ul").classList.toggle('show');
    var attribute = this.parentNode.classList.toggle('show');
    // highlight new position
    new_position = this.hash;
    new_element = document.querySelectorAll('a[href="'+new_position+'"]')[0];
    if (new_element!=current_element) {
        nohl_current(current_element);
        highlight_current(new_element);
    }
    current_element = new_element;
};

// Add listeners to all elements
// Second level list
var classname = document.getElementsByClassName("toctree-l2");
for (var i = 0; i < classname.length; i++) {
    classname[i].querySelector("a").addEventListener('click', toggle_list_show, false);
    if (classname[i].getElementsByTagName('ul').length == 0) {
        classname[i].classList.add('no-expand');
    }
	else if (classname[i].classList.contains('current')) {
        classname[i].classList.add('show');
    }
}

// Remove the doc at the beginning of the name of the functions
var classname = document.getElementsByClassName("toctree-l3");
for (var i = 0; i < classname.length; i++) {
    if (classname[i].classList.contains('current')) {
        classname[i].classList.add('show');
    }
    classname[i].querySelector("a").innerHTML = classname[i].querySelector("a").innerHTML.replace(/.*<\/code>./g, '');
}

