$(document).ready(function(){

    initFields();
    validateFields();

    $('input[type=file][name=dataSetFile]').change(function() {
        validateFields();
    });

    $('input[type=file][name=csvFile]').change(function() {
        
        const csvFile = $('#csvFile').get(0).files[0];
        const reader = new FileReader();
        reader.onload = function (event) {
          const csv = Papa.parse(event.target.result);
          const columns = csv.data[0];
          columns.forEach(column => {
            $('#classColumn').append(new Option(column, column))
            $('#imageIdColumn').append(new Option(column, column))
          });
          console.log("iiiiiiiin")
          $("#classColumn").prop("disabled",false);
          $("#classColumn")[0].selectedIndex = 0;

          $("#imageIdColumn").prop("disabled",false);
          $("#imageIdColumn")[0].selectedIndex = 0;

          validateFields();
        };

        reader.readAsText(csvFile);
    });

    $('#classColumn').change(function() {
        validateFields();
    });

    $('#imageIdColumn').change(function() {
        validateFields();
    });

    $('input[type=radio][name=importMethod]').change(function() {
        const importMethod = $('input[type=radio][name=importMethod]:checked').val();
        // console.log("this.value", this.value)
        if(importMethod === 'folder'){
            $('#csvFile').val('');
            $('#classColumn').val('');
            $('#imageIdColumn').val('');
        }

        validateFields();
        $("#csvInputs").toggleClass("hideCsvFields", (this.value === 'folder'));

    });

    $('input[type=file][name=image]').change(function() {
        validateFields();
    });

    $('input[type=number][name=imagesCount]').change(function() {
        validateFields();
    });

    $('#searchStep').click(function(e) {
        e.preventDefault();
        console.log("Search Completed")
    });
});

const validateFields = () => {
    const importMethod = $('input[type=radio][name=importMethod]:checked').val();
    const hasDataSetFile = $('#dataSetFile').get(0).files.length > 0;
    const hasCsvFile = $('#csvFile').get(0).files.length > 0;
    const classColumn = $('#classColumn').val();
    const imageIdColumn = $('#imageIdColumn').val();
    const modal = $('#modal').val();

    if((hasDataSetFile && modal && importMethod==='folder') || (importMethod==='csv' && hasCsvFile && classColumn && imageIdColumn && modal && hasDataSetFile)){
        $("#nextFirstStep").prop("disabled",false);
        $("#nextFirstStep").hasClass('disabled') && $("#nextFirstStep").removeClass("disabled");
       }else{
        $("#nextFirstStep").prop("disabled",true);
        !$("#nextFirstStep").hasClass('disabled') && $("#nextFirstStep").addClass("disabled");
    }

    const hasImageInput = $('#image').get(0).files.length > 0;
    const imagesCount = $('#imagesCount').val();

    console.log("hasImageInput", hasImageInput)
    console.log("imagesCount", imagesCount)

    if(imagesCount && hasImageInput){
        $("#searchStep").prop("disabled",false);
        $("#searchStep").hasClass('disabled') && $("#searchStep").removeClass("disabled");
    }else{
        $("#searchStep").prop("disabled",true);
        !$("#searchStep").hasClass('disabled') && $("#searchStep").addClass("disabled");
    }
}

const initFields = () => {

    const dataSetFile = $('#dataSetFile').get(0).files.length === 0;
    const csvFile = $('#csvFile').get(0).files.length === 0;
    $("#classColumn").prop("disabled",true);
    $("#imageIdColumn").prop("disabled",true);
    const importMethod = $('input[type=radio][name=importMethod]:checked').val();
    const modal = $('#modal').val();
    console.log("importMethod", importMethod)
    $("#csvInputs").toggleClass("hideCsvFields", (importMethod === 'folder'));

    if(dataSetFile && importMethod && modal){
        $("#nextFirstStep").prop("disabled",false);
        $("#nextFirstStep").hasClass('disabled') && $("#nextFirstStep").removeClass("disabled");
    }else{
        $("#nextFirstStep").prop("disabled",true);
        !$("#nextFirstStep").hasClass('disabled') && $("#nextFirstStep").addClass("disabled");
    }

    const imageInput = $('#image').get(0).files.length === 0;
    const imagesCount = $('#imagesCount').val();
    if(imagesCount && imageInput){
        $("#searchStep").prop("disabled",false);
        $("#searchStep").hasClass('disabled') && $("#searchStep").removeClass("disabled");
    }else{
        $("#searchStep").prop("disabled",true);
        !$("#searchStep").hasClass('disabled') && $("#searchStep").addClass("disabled");
    }
}
