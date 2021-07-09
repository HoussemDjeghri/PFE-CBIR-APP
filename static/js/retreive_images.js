$(document).ready(function(){

    var current_fs, next_fs, previous_fs; //fieldsets
    var opacity;
    var current = 1;
    var steps = $("fieldset").length;

    setProgressBar(current);

    $(".next").click(function(){

        const importMethod = $('input[type=radio][name=importMethod]:checked').val();
        const dataSetFile = $('#dataSetFile').get(0).files[0];
        const modal = $('#modal').val();

        let form = new FormData();
        form.append('dataSet',dataSetFile);
        form.append('modal',modal);
        console.log('text', form.values)
        if(importMethod==='folder'){
        
            $.ajax({
                url: '/extractTestSetFeaturesFolder',
                data: form,
                processData: false,
                contentType: false,
                type: 'POST',
                success: function(response){
                    console.log("response",response)
                    if(response==="success"){
                        goToSecondStep();
                    }else{
                        alert("on error has accured")
                    }
     
                },
                error: function(error){
                    console.log(error);
                }
            });

        }else{
            const csvFile = $('#csvFile').get(0).files[0];
            const classColumn = $('#classColumn').val();
            const imageIdColumn = $('#imageIdColumn').val();

            form.append('csvFile',csvFile);
            form.append('classColumn',classColumn);
            form.append('imageIdColumn',imageIdColumn);

            $.ajax({
                url: '/extractTestSetFeaturesExcel',
                data: form,
                processData: false,
                contentType: false,
                type: 'POST',
                success: function(response){
                    console.log("response",response)
                    if(response==="success"){
                        goToSecondStep();
                    }else{
                        alert("on error has accured")
                    }
                   
                },
                error: function(error){
                    console.log(error);
                }
            });
        }
  
        console.log('hi ajax')
       

    });

    $(".search").click(function(){

        const image = $('#image').get(0).files[0];
        const imagesCount = $('#imagesCount').val();

        let form = new FormData();
        form.append('image',image);
        form.append('imagesCount',imagesCount);

        $.ajax({
            url: '/search',
            data: form,
            processData: false,
            contentType: false,
            type: 'POST',
            success: function(response){
                console.log("response",response)
                // Get a reference to the alert wrapper
                const uploaded_images_wrapper = document.getElementById("uploaded_images_wrapper");
                uploaded_images_wrapper.innerHTML = response.data

            },
            error: function(error){
                console.log(error);
            }
        });
       

    });

    $(".previous").click(function(){

    current_fs = $(this).parent();
    previous_fs = $(this).parent().prev();

    //Remove class active
    $("#progressbar li").eq($("fieldset").index(current_fs)).removeClass("active");

    //show the previous fieldset
    previous_fs.show();

    //hide the current fieldset with style
    current_fs.animate({opacity: 0}, {
    step: function(now) {
    // for making fielset appear animation
    opacity = 1 - now;

    current_fs.css({
    'display': 'none',
    'position': 'relative'
    });
    previous_fs.css({'opacity': opacity});
    },
    duration: 500
    });
    setProgressBar(--current);
    });

    function setProgressBar(curStep){
    var percent = parseFloat(100 / steps) * curStep;
    percent = percent.toFixed();
    $(".progress-bar")
    .css("width",percent+"%")
    }

    $(".submit").click(function(){
    return false;
    })


    const goToSecondStep = () => {
        // /////////////////////////// Loading next page ////////////////
    
                    current_fs = $('.next').parent();
                    next_fs = $('.next').parent().next();
    
                    //Add Class Active
                    $("#progressbar li").eq($("fieldset").index(next_fs)).addClass("active");
    
                    //show the next fieldset
                    next_fs.show();
                    //hide the current fieldset with style
                    current_fs.animate({opacity: 0}, {
                    step: function(now) {
                    // for making fielset appear animation
                    opacity = 1 - now;
    
                    current_fs.css({
                    'display': 'none',
                    'position': 'relative'
                    });
                    next_fs.css({'opacity': opacity});
                    },
                    duration: 500
                    });
                    setProgressBar(++current);
    
                    // /////////////////////////// Loading next page end ////////////////
    }
});